import torch_geometric as pyg
import torch
import torch.nn.functional as F
from torch import nn


class Edge_MLP(torch.nn.Module):
    def __init__(self, vertex_channels, edge_in_channels, edge_out_channels, dropout: float = 0.5, batchnorm_order: str = "post"):
        super().__init__()
        self.vertex_channels = vertex_channels
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels
        self._dropout = dropout
        self._batchnorm_order = batchnorm_order
        self.edge_linear = torch.nn.Linear(2 * self.vertex_channels + self.edge_in_channels, self.edge_out_channels)
        self._bn = nn.BatchNorm1d(self.edge_in_channels if self._batchnorm_order == "pre" else self.edge_out_channels)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        x_i, x_j = x[row], x[col]

        if self._batchnorm_order == "pre":
            edge_attr = self._bn(edge_attr)

        e = self.edge_linear(torch.cat([x_i, x_j, edge_attr], dim=-1))
        if self._batchnorm_order == "post":
            e = self._bn(e)
        e = F.relu(e)
        e = F.dropout(e, p=self._dropout, training=self.training)

        return e


class Edge_EGAT(pyg.nn.MessagePassing):
    def __init__(self, vertex_feature, edge_feature, heads=8, leaky=0.2, dropout=0.1):
        super().__init__()
        # parameters
        self._vertex_feature = vertex_feature
        self._edge_feature = edge_feature
        self._heads = heads
        self._leaky = leaky
        self._dropout = dropout

        # computed
        self._vertex_channel = self._vertex_feature // self._heads
        self._edge_channel = self._edge_feature // self._heads
        self._attention_channel = (self._vertex_feature * 2 + self._edge_feature) // self._heads

        # modules
        self.Wh = nn.Linear(self._vertex_feature, self._vertex_feature)
        self.We = nn.Linear(self._edge_feature, self._edge_feature)
        self.att = nn.Parameter(torch.Tensor(1, self._heads, self._attention_channel))
        self.bias = torch.nn.Parameter(torch.Tensor(self._edge_feature))
        self.mlp_e = nn.Linear(self._edge_feature * 3 + self._vertex_feature * 2, self._edge_feature)
        self.bn = nn.BatchNorm1d(self._edge_feature)

        self.reset_parameters()

    def reset_parameters(self):
        pyg.nn.inits.glorot(self.att)
        pyg.nn.inits.zeros(self.bias)

    def forward(self, h, edge_index, e):
        h = self.Wh(h)
        e = self.We(e)

        new_e = self.propagate(edge_index, h=h, e=e, size=(h.size(0), h.size(0)))

        row, col = edge_index
        e_i, e_j = new_e[row], new_e[col]
        e_ij = e
        h_i, h_j = h[row], h[col]

        # MLP(e_i, e_j, e_ij)
        e = self.mlp_e(torch.cat([e_i, e_j, e_ij, h_i, h_j], dim=-1))
        e = self.bn(e)
        e = F.relu(e)
        e = F.dropout(e, p=self._dropout, training=self.training)

        return e

    def message(self, h_i, h_j, e, size_i, edge_index_i):
        h_j = h_j.view(-1, self._heads, self._vertex_channel)
        h_i = h_i.view(-1, self._heads, self._vertex_channel)
        e = e.view(-1, self._heads, self._edge_channel)
        alpha = (torch.cat([h_i, h_j, e], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self._leaky)
        alpha = pyg.utils.softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self._dropout, training=self.training)

        return e * alpha.view(-1, self._heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self._edge_feature)
        aggr_out = aggr_out + self.bias

        return aggr_out


class Edge_indentity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, edge_attr):
        return edge_attr
