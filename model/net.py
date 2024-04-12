from torch import nn
import torch.nn.functional as F
import torch
import typing as T
from model.node import EGAT_concat, EGAT_merge, EGAT_split
from model.edge import Edge_MLP, Edge_EGAT, Edge_indentity
import model.mgcn as mgcn
import model.nnconv as nnconv
from torch_geometric.nn import GATConv, GCNConv


class ResidualAgg(nn.Module):
    def __init__(self) -> None:
        super(ResidualAgg, self).__init__()

    def forward(self, hidden_value: torch.Tensor, new_value: torch.Tensor) -> T.Tuple[torch.Tensor, torch.Tensor]:
        out = hidden_value + new_value
        return out, out


class GRUAgg(nn.Module):
    def __init__(self, feature: int, dropout) -> None:
        super(GRUAgg, self).__init__()
        self._feature = feature
        self.gru = nn.GRU(self._feature, self._feature)

    def forward(self, hidden_value: torch.Tensor, new_value: torch.Tensor) -> T.Tuple[torch.Tensor, torch.Tensor]:
        x, h = self.gru(new_value.unsqueeze(0), hidden_value.unsqueeze(0))

        return x.squeeze(0), h.squeeze(0)


def agg_module_builder(agg_type: str, **config) -> nn.Module:
    if agg_type == "residual":
        return ResidualAgg()
    elif agg_type == "gru":
        return GRUAgg(
            feature=config.get("feature"),
            dropout=config.get("dropout")
        )
    else:
        raise ValueError("Unknown agg module type: {}".format(agg_type))


class Layer(nn.Module):
    def __init__(self, update_module_vertex=None, update_module_edge=None, **config):
        super(Layer, self).__init__()
        # parameters
        self._vertex_feature = config.get("vertex_feature")  # type: int
        self._edge_feature = config.get("edge_feature")  # type: int
        self._vertex_type = config.get("vertex_type")  # type: str
        self._edge_type = config.get("edge_type")  # type: str
        # self._edge_type = "mgcn" if self._vertex_type in ["mgcn", "mgcn_att"] else "egat"
        self._update_method = config.get("update_method")  # type: str
        self._dropout = config.get("dropout")  # type: float
        assert(self._update_method in ["residual", "gru", "none"])
        self._edge_order = config.get("edge_order")
        assert(self._edge_order in ["before", "after", "parallel"])

        # modules
        self._edge_module = self.edge_module_builder(**config)
        self._vertex_module = self.vertex_module_builder(**config)
        self._update_module_vertex = update_module_vertex
        self._update_module_edge = update_module_edge

    def update_edge(self, x, edge_index, edge_attr, e_hidden=None):
        e = edge_attr
        if e_hidden is None:
            e_hidden = edge_attr

        e = self._edge_module(x, edge_index, e)
        if isinstance(e, tuple):
            e, e_o = e
        else:
            e, e_o = e, e
        e, e_hidden = self._update_module_edge(e_hidden, e) if self._update_method != "none" else (e, None)

        return e, e_o, e_hidden

    def update_vertex(self, x, edge_index, edge_attr, x_hidden=None):
        h, e = x, edge_attr
        if x_hidden is None:
            x_hidden = x

        h = self._vertex_module(h, edge_index, e)
        if isinstance(h, tuple):
            h, h_o = h
        else:
            h, h_o = h, h
        h, x_hidden = self._update_module_vertex(x_hidden, h) if self._update_method != "none" else (h, None)

        return h, h_o, x_hidden

    def forward(self, x, edge_index, edge_attr, x_hidden=None, e_hidden=None):
        if self._edge_order == "before":
            e, e_o, e_hidden = self.update_edge(x, edge_index, edge_attr, e_hidden=e_hidden)
            h, h_o, x_hidden = self.update_vertex(x, edge_index, e, x_hidden=x_hidden)
        elif self._edge_order == "parallel":
            e, e_o, e_hidden = self.update_edge(x, edge_index, edge_attr, e_hidden=e_hidden)
            h, h_o, x_hidden = self.update_vertex(x, edge_index, edge_attr, x_hidden=x_hidden)
        elif self._edge_order == "after":
            h, h_o, x_hidden = self.update_vertex(x, edge_index, edge_attr, x_hidden=x_hidden)
            e, e_o, e_hidden = self.update_edge(h, edge_index, edge_attr, e_hidden=e_hidden)
        return (h, e), (h_o, e_o), (x_hidden, e_hidden)

    @staticmethod
    def edge_module_builder(edge_type: str, **config: T.Union[int, float]) -> nn.Module:
        if edge_type == "mgcn":
            return mgcn.EdgeModule(
                vertex_feature=config.get("vertex_feature"),
                edge_feature=config.get("edge_feature"),
                eta=config.get("eta", 0.8),
                dropout=config.get("dropout", 0.5),
                batchnorm_order=config.get("batchnorm_order", "pre")
            )
        elif edge_type == "mlp":
            return Edge_MLP(
                vertex_channels=config.get("vertex_feature"),
                edge_in_channels=config.get("edge_feature"),
                edge_out_channels=config.get("edge_feature"),
                dropout=config.get("dropout", 0.5),
                batchnorm_order=config.get("batchnorm_order", "pre")
            )
        elif edge_type == "egat":
            return Edge_EGAT(
                vertex_feature=config.get("vertex_feature"),
                edge_feature=config.get("edge_feature"),
                heads=8,
                dropout=config.get("dropout", 0.5)
            )
        elif edge_type == "identity":
            return Edge_indentity()
        else:
            raise ValueError("Unknown edge module type: {}".format(edge_type))

    @staticmethod
    def vertex_module_builder(vertex_type: str, **config: T.Union[int, float]) -> nn.Module:
        if vertex_type == "mgcn_att":
            return mgcn.AttentionVertexModule(
                vertex_feature=config.get("vertex_feature"),
                edge_feature=config.get("edge_feature"),
                dropout=config.get("dropout", 0.5),
                batchnorm_order=config.get("batchnorm_order", "pre"),
                symmetric=config.get("symmetric")
            )
        elif vertex_type == "mgcn":
            return mgcn.VertexModule(
                vertex_feature=config.get("vertex_feature"),
                edge_feature=config.get("edge_feature"),
                dropout=config.get("dropout", 0.5),
                batchnorm_order=config.get("batchnorm_order", "pre")
            )
        elif vertex_type == "egat" or vertex_type == "egat_concat":
            return EGAT_concat(
                vertex_feature=config.get("vertex_feature"),
                edge_feature=config.get("edge_feature"),
                dropout=config.get("dropout", 0.5),
                vertex_feature_ratio=config.get("vertex_feature_ratio", 0.5),
                heads=config.get("heads", 8),
                concat=config.get("concat", True),
                leaky=config.get("leaky", 0.2),
            )
        elif vertex_type == "egat_merge":
            return EGAT_merge(
                vertex_feature=config.get("vertex_feature"),
                edge_feature=config.get("edge_feature"),
                dropout=config.get("dropout", 0.5),
                heads=config.get("heads", 8),
                concat=config.get("concat", True),
                leaky=config.get("leaky", 0.2),
                symmetric=config.get("symmetric")
            )
        elif vertex_type == "egat_split":
            return EGAT_split(
                    vertex_feature=config.get("vertex_feature"),
                    edge_feature=config.get("edge_feature"),
                    dropout=config.get("dropout", 0.5),
                    #vertex_feature_ratio=config.get("vertex_feature_ratio", 0.5),
                    heads=config.get("heads", 8),
                    concat=config.get("concat", True),
                    leaky=config.get("leaky", 0.2)
            )
        elif vertex_type == "nnconv":
            return nnconv.NNConvWrapper(
                vertex_in=config.get("vertex_feature"),
                vertex_out=config.get("vertex_feature"),
                edge_in=config.get("edge_feature"),
                dropout=config.get("dropout", 0.5),
                batchnorm_order=config.get("batchnorm_order", "pre"),
                attention=False
            )
        elif vertex_type == "nnconv_att":
            return nnconv.NNConvWrapper(
                vertex_in=config.get("vertex_feature"),
                vertex_out=config.get("vertex_feature"),
                edge_in=config.get("edge_feature"),
                dropout=config.get("dropout", 0.5),
                batchnorm_order=config.get("batchnorm_order", "pre"),
                attention=True
            )
        else:
            raise ValueError("Unknown vertex module type: {}".format(vertex_type))

    @property
    def vertex_out_out_feature(self):
        return self._vertex_module._vertex_out_out_feature


class AMLSimNet(nn.Module):
    def __init__(self, **config):
        super(AMLSimNet, self).__init__()
        # parameters
        self._vertex_in_feature = config.get("vertex_in_feature")  # type: int
        self._edge_in_feature = config.get("edge_in_feature")  # type: int
        self._num_classes = config.get("num_classes")  # type: int
        self._layers = config.get("layers")  # type: int
        self._vertex_feature = config.get("vertex_feature")  # type: int
        self._edge_feature = config.get("edge_feature")  # type: int
        self._dropout = config.get("dropout")  # type: float
        self._predict_hidden = config.get("predict_hidden", True)  # type: bool
        self._bottleneck = config.get("bottleneck", "linear")  # type: str
        assert(self._bottleneck in ["linear", "egat"])
        self._update_method = config.get("update_method")  # type: str
        assert(self._update_method in ["residual", "gru", "none"])
        self._vertex_type = config.get("vertex_type")
        self._layer_aggregation_method = config.get("layer_aggregation_method")
        assert(self._layer_aggregation_method in ["last", "concat"])
        self._concat_input = config.get("concat_input", True)
        self._batchnorm_order = config.get("batchnorm_order")

        # modules
        self._vertex_in_bn = nn.BatchNorm1d(self._vertex_in_feature)
        self._edge_in_bn = nn.BatchNorm1d(self._edge_in_feature)
        self._vertex_linear = nn.Sequential(
            nn.Linear(self._vertex_in_feature, self._vertex_feature),
            nn.BatchNorm1d(self._vertex_feature) if self._batchnorm_order != "none" else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self._dropout)
        )
        self._edge_linear = nn.Sequential(
            nn.Linear(self._edge_in_feature, self._edge_feature),
            nn.BatchNorm1d(self._edge_feature) if self._batchnorm_order != "none" else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(self._dropout)
        )

        # GNN layers
        if self._update_method != "none":
            self._update_module_vertex = agg_module_builder(self._update_method, feature=self._vertex_feature, dropout=self._dropout)
            self._update_module_edge = agg_module_builder(self._update_method, feature=self._edge_feature, dropout=self._dropout)
        else:
            self._update_module_vertex = None
            self._update_module_edge = None

        self._layer_modules = nn.ModuleList([
            Layer(update_module_vertex=self._update_module_vertex, update_module_edge=self._update_module_edge, **config) for _ in range(self._layers)
        ])

        # layer aggregation
        self._layer_attention_inputs = [self._vertex_feature] + [layer.vertex_out_out_feature for layer in self._layer_modules]

        # predict layer
        if self._predict_hidden:
            self._predict = nn.Sequential(
                nn.Linear(self._predict_feature_in, self._vertex_feature),
                nn.BatchNorm1d(self._vertex_feature) if self._batchnorm_order != "none" else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(p=self._dropout),
                nn.Linear(self._vertex_feature, self._num_classes)
            )
        else:
            self._predict = nn.Linear(self._predict_feature_in, self._num_classes)

    def forward(self, data):
        x, e, e_idx = data.x, data.edge_attr, data.edge_index

        # bottleneck, modify feature dimension
        x = self._vertex_linear(x)
        e = self._edge_linear(e)

        h_outputs = [x]

        if self._update_method != "none":
            x, x_hidden = self._update_module_vertex(x, torch.zeros_like(x))
            e, e_hidden = self._update_module_edge(e, torch.zeros_like(e))
        else:
            x_hidden = None
            e_hidden = None

        # GNN layers
        for layer in self._layer_modules:
            # x_hidden, e_hidden = x, e
            (x, e), (x_o, e_o), (x_hidden, e_hidden) = layer(x, e_idx, e, x_hidden, e_hidden)
            h_outputs.append(x_o)
            # if self._update_method != "none":
            #     x, x_hidden = self._update_module_vertex(x_hidden, x)
            #     e, e_hidden = self._update_module_edge(e_hidden, e)

        # layer aggregation
        if self._layer_aggregation_method == "last":
            h = x
        elif self._layer_aggregation_method == "concat":
            h = torch.cat(h_outputs, dim=-1)
        else:
            raise NotImplementedError

        # classifier
        return self._predict(h)

    @property
    def _predict_feature_in(self):
        if self._layer_aggregation_method == "concat":
            feature = sum(self._layer_attention_inputs)
        else:  # last, attention, lstm
            feature = self._layer_attention_inputs[-1]
        return feature


class CitationNet(nn.Module):
    def __init__(self, **config) -> None:
        super().__init__()
        # parameters
        self._vertex_in_feature = config.get("vertex_in_feature")  # type: int
        self._edge_in_feature = config.get("edge_in_feature")  # type: int
        self._num_classes = config.get("num_classes")  # type: int
        self._vertex_feature = config.get("vertex_feature")  # type: int
        self._edge_feature = config.get("edge_feature")  # type: int
        self._dropout = config.get("dropout")  # type: float
        self._batchnorm = config.get("batchnorm_order") != "none"
        self._vertex_feature_ratio = config.get("vertex_feature_ratio")

        self.method = "egat_concat"  # gcn, gat, gat_egat, egat_concat, egat_merge, egat_split

        if self.method == "gcn":
            self.build_gcn()
        elif self.method == "gat":
            self.build_gat()
        elif self.method == "gat_egat":
            self.build_gat_egat()
        elif self.method == "egat_concat":
            self.build_egat_concat()
            self.build_edge()
        elif self.method == "egat_merge":
            self.build_egat_merge()
            self.build_edge()
        elif self.method == "egat_split":
            self.build_egat_split()
            self.build_edge()

    def build_gat(self):
        self.conv1 = GATConv(self._vertex_in_feature, self._vertex_feature // 8, heads=8, concat=True, dropout=self._dropout)
        self.conv2 = GATConv(self._vertex_feature, self._vertex_feature // 8, heads=8, concat=True, dropout=self._dropout)
        self.conv3 = GATConv((self._vertex_feature), self._num_classes, heads=8, concat=False, dropout=self._dropout)
        self.bn1 = nn.BatchNorm1d(self._vertex_feature) if self._batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(self._vertex_feature) if self._batchnorm else nn.Identity()

    def build_gcn(self):
        self.conv1 = GCNConv(self._vertex_in_feature, self._vertex_feature)
        self.conv2 = GCNConv(self._vertex_feature, self._vertex_feature)
        self.conv3 = GCNConv(self._vertex_feature, self._num_classes)
        self.bn1 = nn.BatchNorm1d(self._vertex_feature) if self._batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(self._vertex_feature) if self._batchnorm else nn.Identity()

    def build_gat_egat(self):
        self.egat = EGAT_concat(self._vertex_feature, self._edge_feature, vertex_feature_ratio=self._vertex_feature_ratio, vertex_in_feature=self._vertex_in_feature, edge_in_feature=self._edge_in_feature, heads=8, concat=True, dropout=self._dropout, batchnorm=self._batchnorm)
        self.conv2 = GCNConv(self._vertex_feature, self._vertex_feature)
        self.conv3 = GCNConv(self._vertex_feature, self._num_classes)
        self.bn1 = nn.BatchNorm1d(self._vertex_feature) if self._batchnorm else nn.Identity()
        self.bn2 = nn.BatchNorm1d(self._vertex_feature) if self._batchnorm else nn.Identity()

    def build_egat_concat(self):
        self.conv1 = EGAT_concat(self._vertex_feature, self._edge_feature, heads=8, vertex_in_feature=self._vertex_in_feature, vertex_feature_ratio=self._vertex_feature_ratio, dropout=self._dropout, batchnorm=self._batchnorm)
        self.conv2 = EGAT_concat(self._vertex_feature, self._edge_feature, heads=8, vertex_feature_ratio=self._vertex_feature_ratio, dropout=self._dropout, batchnorm=self._batchnorm)
        self.conv3 = GATConv((self._vertex_feature), self._num_classes, heads=8, concat=False, dropout=self._dropout)

    def build_egat_merge(self):
        self.conv1 = EGAT_merge(self._vertex_in_feature, self._edge_feature, heads=8, vertex_out_feature=self._vertex_feature, dropout=self._dropout, batchnorm=self._batchnorm)
        self.conv2 = EGAT_merge(self._vertex_feature, self._edge_feature, heads=8, dropout=self._dropout, batchnorm=self._batchnorm)
        self.conv3 = GATConv((self._vertex_feature), self._num_classes, heads=8, concat=False, dropout=self._dropout)

    def build_egat_split(self):
        self.conv1 = EGAT_split(self._vertex_feature, self._edge_feature, self._vertex_in_feature, heads=8, dropout=self._dropout, batchnorm=self._batchnorm)
        self.conv2 = EGAT_split(self._vertex_feature, self._edge_feature, heads=8, dropout=self._dropout, batchnorm=self._batchnorm)
        self.conv3 = GATConv((self._vertex_feature + self._edge_feature), self._num_classes, heads=8, concat=False, dropout=self._dropout)

    def build_edge(self):
        self.edge1 = Edge_MLP(self._vertex_in_feature, self._edge_in_feature, self._edge_feature, dropout=self._dropout)
        self.edge2 = Edge_MLP(self._vertex_feature, self._edge_feature, self._edge_feature, dropout=self._dropout)

    def forward(self, data):
        if self.method in ["gcn", "gat"]:
            return self.forward_gat(data)
        elif self.method == "gat_egat":
            return self.forward_egat_gat(data)
        elif self.method in ["egat_concat", "egat_merge"]:
            return self.forward_egat(data)
        elif self.method == "egat_split":
            return self.forward_egat_split(data)

    def forward_gat(self, data):
        x = data.x
        x = self.conv1(x, data.edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)
        # x = self.conv2(x, data.edge_index)
        # x = self.bn2(x)
        # x = F.elu(x)
        # x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.conv3(x, data.edge_index)

        return x

    def forward_egat(self, data):
        x = data.x
        e = self.edge1(x, data.edge_index, data.edge_attr)
        x = self.conv1(x, data.edge_index, e)
        e = self.edge2(x, data.edge_index, e)
        x = self.conv2(x, data.edge_index, e)
        x = self.conv3(x, data.edge_index)

        return x

    def forward_egat_gat(self, data):
        x = self.egat(data.x, data.edge_index, data.edge_attr)
        x = self.conv2(x, data.edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self._dropout, training=self.training)
        x = self.conv3(x, data.edge_index)

        return x

    def forward_egat_split(self, data):
        x = data.x
        e = self.edge1(x, data.edge_index, data.edge_attr)
        x, x_out = self.conv1(x, data.edge_index, e)
        e = self.edge2(x, data.edge_index, e)
        x, x_out = self.conv2(x, data.edge_index, e)
        x = self.conv3(x_out, data.edge_index)

        return x
