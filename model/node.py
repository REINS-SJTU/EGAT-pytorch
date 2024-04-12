import torch_geometric as pyg
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class EGAT_base(MessagePassing):
    def __init__(self, vertex_feature, edge_feature, heads=1, concat=True,
                 leaky=0.2, dropout=0, skip=False, batchnorm=True, **kwargs):
        super(EGAT_base, self).__init__(aggr='add', **kwargs)
        # parameters
        self._heads = heads
        self._vertex_feature = vertex_feature
        self._edge_feature = edge_feature
        self._concat = concat
        self._leaky = leaky
        self._dropout = dropout
        self._apart = False
        self._skip = skip
        self._batchnorm = batchnorm
        self.ptr=None

    def reset_parameters(self):
        pyg.nn.inits.glorot(self.att)

    def build_modules(self):
        # attention modules
        self.att = torch.nn.Parameter(torch.Tensor(1, self._heads, self._m_att_channel))
        self.bn = nn.BatchNorm1d(self._vertex_out_feature) if self._batchnorm else nn.Identity()
        self.build_custom_modules()
        self.reset_parameters()

    def build_custom_modules(self):
        pass

    def custom_update(self, x_i, aggr_out):
        return aggr_out

    def forward(self, x, edge_index, edge_attr):
        h = self.Vtransform(x)
        e = self.Etransform(edge_attr)

        h = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=h, e=e, x_o=x, e_o=edge_attr)


        if self._apart:
            h, h_out = h[:, :self._vertex_out_feature], h[:, self._vertex_out_feature:]
        if not self._skip:
            h = self.bn(h)
            h = F.elu(h)
            h = F.dropout(h, p=self._dropout, training=self.training)

        if self._apart:
            h_out = self.apart_out_process(h_out)
            return h, h_out
        else:
            return h

    def message(self, edge_index_i, x_i, x_j, size_i, e, x_o_i, x_o_j, e_o, index):
        e_size = e.size(0)
        result = self.M(x_i, x_j, e, x_o_i, x_o_j, e_o)
        if self._apart:
            m, m_out, Mtmp = result
        else:
            m, m_out, Mtmp = result[0], result[0], result[1]
        m_att = self.Matt(x_i, x_j, e, x_o_i, x_o_j, e_o, Mtmp)

        m = m.view(e_size, self._heads, -1)
        alpha = self.attention(m_att, edge_index_i, size_i, index, self.ptr)

        if self._apart:
            m, m_out = m * alpha.view(-1, self._heads, 1), m_out * alpha.view(-1, self._heads, 1)
            return torch.cat([m, m_out], dim=-1)
        else:
            return (m * alpha.view(-1, self._heads, 1)).view(e_size, -1)

    def update(self, aggr_out, x):
        size = aggr_out.size(0)
        if self._concat is True:
            aggr_out = aggr_out.view(size, -1)
        else:
            aggr_out = aggr_out.mean(dim=1)

        aggr_out = self.custom_update(x, aggr_out)

        return aggr_out

    def attention(self, m_att, edge_index_i, size_i, index, ptr):
        alpha = (m_att * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self._leaky)
        alpha = pyg.utils.softmax(alpha, index, ptr, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self._dropout, training=self.training)

        return alpha

    def M(self, x_i, x_j, e_ij, x_o_i, x_o_j, e_o):
        size = x_j.size(0)
        x_j = x_j.view((size, self._heads, -1))
        e_ij = e_ij.view((size, self._heads, -1))
        return torch.cat([x_j, e_ij], dim=-1), torch.cat([x_j, e_ij], dim=-1), None

    def Matt(self, x_i, x_j, e_ij, x_o_i, x_o_j, e_o, Mtmp=None):
        size = x_j.size(0)
        x_i = x_i.view((size, self._heads, -1))
        x_j = x_j.view((size, self._heads, -1))
        e_ij = e_ij.view((size, self._heads, -1))
        return torch.cat([x_i, x_j, e_ij], dim=-1)

    def Vtransform(self, x):
        return x

    def Etransform(self, e):
        return e

    def apart_out_process(self, h_out):
        raise NotImplementedError

    @property
    def _vertex_out_feature(self):
        return self._vertex_feature

    @property
    def _m_att_channel(self):
        return self._vertex_out_feature // self._heads if self._concat else self._vertex_out_feature

    @property
    def _vertex_out_out_feature(self):
        return self._vertex_out_feature


class EGAT_concat(EGAT_base):


    def __init__(self, vertex_feature, edge_feature, vertex_feature_ratio, vertex_in_feature=None, edge_in_feature=None, heads=1, concat=True,
                 leaky=0.2, dropout=0, **kwargs):
        # parameter
        super(EGAT_concat, self).__init__(vertex_feature, edge_feature, heads, concat, leaky, dropout, **kwargs)
        self._vertex_feature_ratio = vertex_feature_ratio
        self._vertex_hidden_feature = int(self._vertex_feature_ratio * self._vertex_out_feature)
        self._edge_hidden_feature = self._vertex_out_feature - self._vertex_hidden_feature
        self._vertex_in_feature = vertex_in_feature if vertex_in_feature is not None else vertex_feature
        self._edge_in_feature = edge_in_feature if edge_in_feature is not None else edge_feature
        self.build_modules()

    def build_custom_modules(self):
        self.v_linear = nn.Linear(self._vertex_in_feature, self._vertex_hidden_feature, bias=False)
        self.e_linear = nn.Linear(self._edge_in_feature, self._edge_hidden_feature, bias=False)

    def reset_parameters(self):
        EGAT_base.reset_parameters(self)
        pyg.nn.inits.glorot(self.v_linear.weight)
        pyg.nn.inits.glorot(self.e_linear.weight)

    def M(self, x_i, x_j, e_ij, x_o_i, x_o_j, e_o):
        size = x_j.size(0)
        x_j = x_j.view((size, self._heads, -1))
        e_ij = e_ij.view((size, self._heads, -1))
        return torch.cat([x_j, e_ij], dim=-1), None

    def Matt(self, x_i, x_j, e_ij, x_o_i, x_o_j, e_o, Mtmp=None):
        size = x_j.size(0)
        x_i = x_i.view((size, self._heads, -1))
        x_j = x_j.view((size, self._heads, -1))
        e_ij = e_ij.view((size, self._heads, -1))
        return torch.cat([x_i, x_j, e_ij], dim=-1)

    def Vtransform(self, x):
        return self.v_linear(x)

    def Etransform(self, e):
        return self.e_linear(e)

    def custom_update(self, x_i, aggr_out):
        return aggr_out

    @property
    def _vertex_channel(self):
        return self._vertex_hidden_feature // self._heads

    @property
    def _m_att_channel(self):
        return (self._vertex_hidden_feature * 2 + self._edge_hidden_feature) // self._heads


class EGAT_merge(EGAT_base):
    def __init__(self, vertex_feature, edge_feature, vertex_out_feature=None, heads=1, concat=True,
                 leaky=0.2, dropout=0, symmetric=False, **kwargs):
        # parameter
        self.__vertex_out_feature = vertex_out_feature if vertex_out_feature is not None else vertex_feature
        self._symmetric = symmetric
        super(EGAT_merge, self).__init__(vertex_feature, edge_feature, heads, concat, leaky, dropout, **kwargs)
        self.build_modules()

    def build_custom_modules(self):
        self.v_linear = nn.Linear(self._vertex_feature, self._vertex_out_feature, bias=False)
        self.linear = nn.Linear(self._vertex_feature + self._edge_feature, self._vertex_channel * self._heads, bias=False)
        self.bias = torch.nn.Parameter(torch.Tensor(self._vertex_out_feature))

    def reset_parameters(self):
        EGAT_base.reset_parameters(self)
        pyg.nn.inits.zeros(self.bias)

    def Vtransform(self, x):
        return x

    def M(self, x_i, x_j, e_ij, x_o_i, x_o_j, e_o):
        size = x_j.size(0)
        m = torch.cat([x_o_j, e_ij], dim=-1)
        m = self.linear(m)
        m = m.view((size, self._heads, -1))

        return m, m

    def Matt(self, x_i, x_j, e_ij, x_o_i, x_o_j, e_o, Mtmp=None):
        size = x_j.size(0)
        if self._symmetric:
            m = torch.cat([x_o_i, e_ij], dim=-1)
            m = self.linear(m)
            x_i = m.view((size, self._heads, -1))
        else:
            x_i = x_i.view((size, self._heads, -1))
        return torch.cat([x_i, Mtmp], dim=-1)

    def Etransform(self, e):
        return e

    def custom_update(self, x, aggr_out):
        return aggr_out

    @property
    def _vertex_channel(self):
        return self._vertex_out_feature // self._heads if self._concat else self._vertex_out_feature

    @property
    def _m_att_channel(self):
        return (self._vertex_out_feature + self._vertex_out_feature) // self._heads if self._concat else (self._vertex_feature // self._heads + self._vertex_out_feature)

    @property
    def _vertex_out_feature(self):
        return self.__vertex_out_feature


class EGAT_split(EGAT_base):
    def __init__(self, vertex_feature, edge_feature, vertex_in_feature=None, edge_in_feature=None, heads=1, concat=True,
                 leaky=0.2, dropout=0, batchnorm=True, **kwargs):
        # parameter
        super(EGAT_split, self).__init__(vertex_feature, edge_feature, heads, concat, leaky, dropout, batchnorm=batchnorm, **kwargs)

        self._apart = True
        self._vertex_in_feature = vertex_in_feature if vertex_in_feature is not None else vertex_feature
        self._edge_in_feature = edge_in_feature if edge_in_feature is not None else edge_feature
        self.build_modules()

    def build_custom_modules(self):
        self.v_linear = nn.Linear(self._vertex_in_feature, self._vertex_feature, bias=False)
        self.e_linear = nn.Linear(self._edge_in_feature, self._edge_feature, bias=False)
        self.out_bn = nn.BatchNorm1d(self._vertex_feature + self._edge_feature) if self._batchnorm else nn.Identity()

    def reset_parameters(self):
        EGAT_base.reset_parameters(self)
        pyg.nn.inits.glorot(self.v_linear.weight)
        pyg.nn.inits.glorot(self.e_linear.weight)


    def M(self, x_i, x_j, e_ij, x_o_i, x_o_j, e_o):
        size = x_j.size(0)
        x_i = x_i.view((size, self._heads, -1))
        x_j = x_j.view((size, self._heads, -1))
        e_ij = e_ij.view((size, self._heads, -1))
        return torch.cat([x_j], dim=-1), torch.cat([e_ij, x_j], dim=-1), None

    def Matt(self, x_i, x_j, e_ij, x_o_i, x_o_j, e_o, Mtmp=None):
        size = x_j.size(0)
        x_i = x_i.view((size, self._heads, -1))
        x_j = x_j.view((size, self._heads, -1))
        e_ij = e_ij.view((size, self._heads, -1))
        return torch.cat([x_i, x_j, e_ij], dim=-1)

    def Vtransform(self, x):
        return self.v_linear(x)

    def Etransform(self, e):
        return self.e_linear(e)

    def apart_out_process(self, h_out):
        h_out = self.out_bn(h_out)
        h_out = F.elu(h_out)
        h_out = F.dropout(h_out, p=self._dropout, training=self.training)

        return h_out

    @property
    def _vertex_channel(self):
        return self._vertex_feature // self._heads

    @property
    def _m_att_channel(self):
        return (self._vertex_feature * 2 + self._edge_feature) // self._heads

    @property
    def _vertex_out_out_feature(self):
        return self._vertex_feature + self._edge_feature
