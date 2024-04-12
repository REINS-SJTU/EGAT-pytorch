import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
from torch import nn

'''
https://arxiv.org/pdf/1906.11081.pdf
'''


class EdgeModule(nn.Module):
    '''
    Eq.5.
    '''
    def __init__(self, vertex_feature: int, edge_feature: int, eta: float = 0.8, dropout: float = 0.5, batchnorm_order: str = "pre"):
        super(EdgeModule, self).__init__()
        # parameters
        self._eta = eta
        self._vertex_feature = vertex_feature
        self._edge_feature = edge_feature
        self._dropout = dropout
        self._batchnorm_order = batchnorm_order
        assert(self._batchnorm_order in ["pre", "post", "none"])
        # modules
        self._Wue = nn.Linear(self._vertex_feature, self._edge_feature, bias=False)
        self._bn = nn.BatchNorm1d(self._edge_feature)
        self._vbn = nn.BatchNorm1d(self._vertex_feature)

    def forward(self, x, edge_index, edge_attr):
        if self._batchnorm_order == "pre":
            edge_attr = self._bn(edge_attr)
            x = self._vbn(x)

        row, col = edge_index
        x_i, x_j = x[row], x[col]
        e = self._eta * edge_attr + (1 - self._eta) * self._Wue(x_i * x_j)
        if self._batchnorm_order == "post":
            e = self._bn(e)
        e = F.dropout(e, p=self._dropout, training=self.training)
        return e


class VertexModule(MessagePassing):
    '''
    Eq.6.
    '''
    def __init__(self, vertex_feature: int, edge_feature: int, dropout: float = 0.5, batchnorm_order: str = "pre"):
        super(VertexModule, self).__init__()
        # parameters
        self._vertex_feature = vertex_feature
        self._edge_feature = edge_feature
        self._dropout = dropout
        self._batchnorm_order = batchnorm_order
        assert(self._batchnorm_order in ["pre", "post", "none"])
        # modules
        self._Mfa = nn.Linear(vertex_feature, vertex_feature)
        self._Mfe = nn.Linear(edge_feature, vertex_feature)
        self._Wuv = nn.Linear(vertex_feature, vertex_feature, bias=False)
        self._act = nn.Tanh()
        self._bn = nn.BatchNorm1d(self._vertex_feature)
        self._ebn = nn.BatchNorm1d(self._edge_feature)

    def forward(self, x, edge_index, edge_attr, rbf=None):
        if self._batchnorm_order == "pre":
            x = self._bn(x)
            edge_attr = self._ebn(edge_attr)

        edge_attr = self._Mfe(edge_attr)
        x = self._Mfa(x)

        h = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, edge_attr=edge_attr)
        if self._batchnorm_order == "post":
            h = self._bn(h)
        h = self._act(h)
        h = F.dropout(h, p=self._dropout, training=self.training)

        return h

    def message(self, x_j, edge_attr):
        h = x_j + edge_attr
        h = self._Wuv(h)

        return h

    def update(self, aggr_out, x):
        return aggr_out

    @property
    def _vertex_out_out_feature(self):
        return self._vertex_feature


class AttentionVertexModule(VertexModule):
    def __init__(self, vertex_feature: int, edge_feature: int, dropout: float = 0.5, batchnorm_order: str = "pre", symmetric=False) -> None:
        # parameters
        self._vertex_feature = vertex_feature
        self._edge_feature = edge_feature
        self._dropout = dropout
        self._batchnorm_order = batchnorm_order
        assert(self._batchnorm_order in ["pre", "post", "none"])
        self._heads = 8
        self._channels = self._vertex_feature // self._heads
        self._symmetric = symmetric

        super(AttentionVertexModule, self).__init__(
            vertex_feature=self._vertex_feature,
            edge_feature=self._edge_feature,
            dropout=self._dropout,
            batchnorm_order=self._batchnorm_order
        )
        self.att = torch.nn.Parameter(torch.Tensor(1, self._heads, self._channels * 2))
        self.bias = torch.nn.Parameter(torch.Tensor(self._vertex_feature))

        self.reset_parameters2()

    def reset_parameters2(self):
        pyg.nn.inits.glorot(self.att)
        pyg.nn.inits.zeros(self.bias)

    def message(self, x_i, x_j, edge_attr, edge_index_i, size_i, index, ptr):
        x_j = super(AttentionVertexModule, self).message(x_j, edge_attr)
        if self._symmetric:
            x_i = super(AttentionVertexModule, self).message(x_i, edge_attr)

        x_j = x_j.view(-1, self._heads, self._channels)
        x_i = x_i.view(-1, self._heads, self._channels)

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        # alpha = pyg.utils.softmax(alpha, edge_index_i, size_i)
        alpha = pyg.utils.softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self._dropout, training=self.training)

        return (x_j * alpha.view(-1, self._heads, 1)).view(-1, self._vertex_feature) + self.bias

    @property
    def _vertex_out_out_feature(self):
        return self._vertex_feature
