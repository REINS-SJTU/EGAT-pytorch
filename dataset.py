import torch_geometric as pyg
import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset
import networkx as nx
import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class AMLSimDataset(InMemoryDataset):
    def __init__(self, root, mode='undirected', transform=None, pre_transform=None):
        self.mode = mode
        super().__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def _root(self, root, name):
        return root

    def data_process(self, g):
        if self.mode == "concat":
            return g.x.float(), torch.cat([g.edge_attr, g.r_edge_attr, g.edge_attr + g.r_edge_attr], dim=1).float()
        else:
            return g.x.float(), g.edge_attr.float()

    @property
    def raw_file_names(self):
        return ['labels.npy', 'labeled_nodes.npy', 'idx_train.npy', 'idx_val.npy', 'idx_test.npy', 'graph.pkl']

    @property
    def processed_file_names(self):
        if self.mode == "undirected":
            return ["data_u.pt"]
        elif self.mode == "directed":
            return ["data_d.pt"]
        elif self.mode == "concat":
            return ["data_c.pt"]
        else:
            raise "Unknown mode: {}".format(self.mode)

    def _torch_load(self, filename):
        fullname = os.path.join(self.raw_dir, filename)
        f = open(fullname, "rb")
        data = np.load(f)
        return torch.from_numpy(data)

    @staticmethod
    def _reverse_dict(d: dict):
        r_d = {}
        for k in d:
            if not k.startswith('r_'):
                r_d["r_{}".format(k)] = d[k]

        return r_d

    @staticmethod
    def _empty_reverse_dict(d: dict):
        r_d = {}
        for k in d:
            if not k.startswith('r_'):
                if isinstance(d[k], list):
                    r_d["r_{}".format(k)] = [0 for _ in range(len(d[k]))]
                else:
                    r_d["r_{}".format(k)] = 0

        return r_d

    @staticmethod
    def _empty_dict(d: dict):
        r_d = {}
        for k in d:
            if not k.startswith('r_'):
                if isinstance(d[k], list):
                    r_d["{}".format(k)] = [0 for _ in range(len(d[k]))]
                else:
                    r_d["{}".format(k)] = 0

        return r_d

    @staticmethod
    def _edge_feature_concat(g: nx.DiGraph):
        for u, nbrs in g._adj.items():
            for v, d in nbrs.items():
                if (v, u) in g.edges:
                    g.add_edge(v, u, **AMLSimDataset._reverse_dict(d))
                else:
                    g.add_edge(v, u, **AMLSimDataset._reverse_dict(d), **AMLSimDataset._empty_dict(d))
                    g.add_edge(u, v, **AMLSimDataset._empty_reverse_dict(d))
        return g

    def download(self):
        for f in self.raw_paths:
            if not os.path.exists(f):
                raise Exception("File not found: {}".format(f))

    def process(self):
        train_idx = self._torch_load("idx_train.npy")
        val_idx = self._torch_load("idx_val.npy")
        test_idx = self._torch_load("idx_test.npy")
        labeled_nodes = self._torch_load("labeled_nodes.npy")
        labels = self._torch_load("labels.npy")

        graph: nx.DiGraph = pkl.load(open(os.path.join(self.raw_dir, "graph.pkl"), "rb"))

        num_nodes = graph.number_of_nodes()
        y = torch.zeros((num_nodes,)).long()
        for i in range(labels.max() + 1):
            pos_node_idx = labeled_nodes[labels == i]
            y[pos_node_idx] = i

        train_mask = torch.zeros((num_nodes,)).bool()
        train_mask[train_idx] = True
        val_mask = torch.zeros((num_nodes,)).bool()
        val_mask[val_idx] = True
        test_mask = torch.zeros((num_nodes,)).bool()
        test_mask[test_idx] = True

        if self.mode == "undirected":
            graph = graph.to_undirected()
        elif self.mode == "concat":
            graph = AMLSimDataset._edge_feature_concat(graph)
        raw_g = pyg.utils.from_networkx(graph)

        x, edge_attr = self.data_process(raw_g)

        g = pyg.data.Data(x=x, edge_index=raw_g.edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        # Read data into huge `Data` list.
        data_list = [g]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BatchAMLSimDataset(AMLSimDataset):
    def __init__(self, root, mode='concat', transform=None, pre_transform=None):
        super().__init__(root, mode=mode, transform=transform, pre_transform=pre_transform)

    def process(self):
        data_list = []

        for i in tqdm(range(100)):
            try:
                data_list.append(self.process_one(i))
            except Exception as e:
                print(e)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def process_one(self, idx):
        train_idx = self._torch_load(idx, "idx_train.npy")
        val_idx = self._torch_load(idx, "idx_val.npy")
        test_idx = self._torch_load(idx, "idx_test.npy")
        labeled_nodes = self._torch_load(idx, "labeled_nodes.npy")
        labels = self._torch_load(idx, "labels.npy")

        graph: nx.DiGraph = pkl.load(open(os.path.join(self.raw_dir, str(idx), "graph.pkl"), "rb"))

        num_nodes = graph.number_of_nodes()
        y = torch.zeros((num_nodes,)).long()
        for i in range(labels.max() + 1):
            pos_node_idx = labeled_nodes[labels == i]
            y[pos_node_idx] = i

        train_mask = torch.zeros((num_nodes,)).bool()
        train_mask[train_idx] = True
        val_mask = torch.zeros((num_nodes,)).bool()
        val_mask[val_idx] = True
        test_mask = torch.zeros((num_nodes,)).bool()
        test_mask[test_idx] = True

        if self.mode == "undirected":
            graph = graph.to_undirected()
        elif self.mode == "concat":
            graph = AMLSimDataset._edge_feature_concat(graph)
        raw_g = pyg.utils.from_networkx(graph)

        x, edge_attr = self.data_process(raw_g)
        edge_label, _ = y[raw_g.edge_index].max(dim=0)

        g = pyg.data.Data(x=x, edge_index=raw_g.edge_index, edge_attr=edge_attr, edge_label=edge_label, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        return g

    def _torch_load(self, idx, filename):
        fullname = os.path.join(self.raw_dir, str(idx), filename)
        f=open(fullname, "rb")
        data = np.load(f)
        return torch.from_numpy(data)


class Cora(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None) -> None:
        self._root = root
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["cora.cites", "cora.content"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        id_map, x, y = self.read_nodes()
        edge_index = self.read_graph(id_map)

        g = pyg.data.Data(x=x, y=y, edge_index=edge_index)

        data_list = [g]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def read_graph(self, id_map: dict):
        filename = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(filename, "r") as f:
            lines = f.readlines()
            result = [self.process_line_cites(line, id_map) for line in lines]
            result = filter(lambda x: x is not None, result)
            row, col = list(map(list, list(zip(*result))))
            edge_index = torch.Tensor([row, col]).long()

        return edge_index

    def read_nodes(self):
        filename = os.path.join(self.raw_dir, self.raw_file_names[1])
        with open(filename, "r") as f:
            lines = f.readlines()
            result = [self.process_line_content(line) for line in lines]
            node_ids, node_classes, node_features = list(map(list, list(zip(*result))))
            id_map = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
            y = torch.Tensor(self.encode_label(node_classes)).long()
            x = torch.Tensor(node_features)

        return id_map, x, y

    def process_line_content(self, line: str):
        data = line.split("\t")
        node_id = data[0]
        node_class = data[-1][:-1]
        node_feature = [int(i) for i in data[1: -1]]

        return node_id, node_class, node_feature

    def process_line_cites(self, line: str, id_map: dict):
        data = line[:-1].split("\t")

        try:
            return id_map[data[0]], id_map[data[1]]
        except KeyError:
            print("KeyError: {} - {}".format(data[0], data[1]))
            return None

    def encode_label(self, labels):
        le = LabelEncoder()
        return le.fit_transform(labels)


class CiteSeer(Cora):
    @property
    def raw_file_names(self):
        return ["citeseer.cites", "citeseer.content"]

    def process(self):
        super().process()


class Pubmed(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None) -> None:
        self._root = root
        super().__init__(root=root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["Pubmed-Diabetes.DIRECTED.cites.tab", "Pubmed-Diabetes.NODE.paper.tab"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        id_map, x, y = self.read_nodes()
        edge_index = self.read_graph(id_map)

        g = pyg.data.Data(x=x, y=y, edge_index=edge_index)

        data_list = [g]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def read_nodes(self):
        filename = os.path.join(self.raw_dir, self.raw_file_names[1])
        id_map: dict[str, int] = {}
        labels: list[int] = []
        features: list[list[float]] = []

        with open(filename, "r") as f:
            f.readline()
            self.words_list = self.parse_wordlist(f.readline())
            for (i, line) in enumerate(f.readlines()):
                node_id, label, feature = self.parse_nodedata(line)
                id_map[node_id] = i
                labels.append(label)
                features.append(feature)

        return id_map, torch.Tensor(features), torch.Tensor(labels).long()

    def read_graph(self, id_map):
        filename = os.path.join(self.raw_dir, self.raw_file_names[0])
        with open(filename, "r") as f:
            f.readline()
            f.readline()  # skip 2 lines
            lines = f.readlines()
            result = [self.parse_edge(line, id_map) for line in lines]
            result = filter(lambda x: x is not None, result)
            row, col = list(map(list, list(zip(*result))))
            edge_index = torch.Tensor([row, col]).long()

        return edge_index

    def parse_wordlist(self, line):
        data = line.split("\t")[1: -1]

        def get_word(text):
            return text.split(":")[1]

        words = list(map(get_word, data))

        return words

    def parse_nodedata(self, line):
        data = line.split("\t")[: -1]
        node_id = data[0]
        label = int(data[1].split("=")[1])
        words = data[2:]

        words_dict: dict[str, float] = dict({(w, 0.0) for w in self.words_list})
        for word_pair in words:
            word, value = word_pair.split("=")
            words_dict[word] = float(value)

        feature = [words_dict[w] for w in self.words_list]

        return node_id, label, feature

    def parse_edge(self, line, id_map):
        data = line[: -1].split("\t")
        row = data[1][6:]
        col = data[3][6:]

        try:
            return id_map[row], id_map[col]
        except KeyError:
            print("KeyError: {} - {}".format(row, col))
            return None
