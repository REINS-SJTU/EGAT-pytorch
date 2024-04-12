import torch_geometric as pyg
from torch_geometric.nn import NNConv
import torch
from torch import nn
import torch.nn.functional as F


class NNConvWrapper(nn.Module):
    def __init__(self, vertex_in, vertex_out, edge_in, batchnorm_order, attention, dropout) -> None:
        super(NNConvWrapper, self).__init__()
        self._vertex_in = vertex_in
        self._vertex_out = vertex_out
        self._edge_in = edge_in
        self._hidden = 128
        self._dropout = dropout
        self._batchnorm_order = batchnorm_order
        assert(self._batchnorm_order in ["pre", "post", "none"])
        self._attention = attention

        self.nn = nn.Sequential(
            nn.Linear(self._edge_in, self._hidden),
            nn.ReLU(),
            nn.Linear(self._hidden, self._vertex_in * self._vertex_out)
        )

        if self._attention:
            self.conv = AttentionNNConv(self._vertex_in, self._vertex_out, self.nn, dropout=self._dropout)
        else:
            self.conv = NNConv(self._vertex_in, self._vertex_out, self.nn, aggr="mean")
        self.bn_v = nn.BatchNorm1d(self._vertex_in)
        self.bn_e = nn.BatchNorm1d(self._edge_in)

    def forward(self, x, e_idx, e):

        if self._batchnorm_order == "pre":
            x = self.bn_v(x)
            e = self.bn_e(e)
        x = self.conv(x, e_idx, e)
        if self._batchnorm_order == "post":
            x = self.bn_v(x)
        x = F.elu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)

        return x

    @property
    def _vertex_out_out_feature(self):
        return self._vertex_out


class AttentionNNConv(NNConv):
    def __init__(self, vertex_in, vertex_out, nn, dropout, **kwargs) -> None:
        self._vertex_in = vertex_in
        self._vertex_out = vertex_out
        self._dropout = dropout
        self._heads = 8
        self._channels = self._vertex_out // self._heads
        super(AttentionNNConv, self).__init__(self._vertex_in, self._vertex_out, nn, bias=False, **kwargs)
        self.att = torch.nn.Parameter(torch.Tensor(1, self._heads, self._channels * 2))
        self.bias = torch.nn.Parameter(torch.Tensor(self._vertex_out))
        self.bn = torch.nn.BatchNorm1d(self._heads)

        self.reset_parameters2()

    def reset_parameters2(self):
        super(AttentionNNConv, self).reset_parameters()
        pyg.nn.inits.glorot(self.att)
        pyg.nn.inits.zeros(self.bias)

    def message(self, x_i, x_j, edge_attr, edge_index_i, size_i, ptr, index):
        weight = self.nn(edge_attr).view(-1, self.in_channels, self.out_channels)
        x_j = torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)
        x_i = torch.matmul(x_i.unsqueeze(1), weight).squeeze(1)

        x_j = x_j.view(-1, self._heads, self._channels)
        x_i = x_i.view(-1, self._heads, self._channels)

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        # alpha = self.bn(alpha.squeeze(-1)).unsqueeze(-1)
        alpha = F.leaky_relu(alpha, 0.2)
        # alpha = pyg.utils.softmax(alpha, edge_index_i, size_i)
        alpha = pyg.utils.softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self._dropout, training=self.training)

        return (x_j * alpha.view(-1, self._heads, 1)).view(-1, self._vertex_out) + self.bias
