import torch
import numpy as np
from torch_scatter import scatter_max, scatter_mean, scatter_std, scatter_min


class AttachEdgeAttr(object):
    def __call__(self, data):
        row, col = data.edge_index
        x = data.x
        x_i, x_j = x[row], x[col]
        data.edge_attr = (x_i * x_j).sum(dim=-1, keepdim=True)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


class AddSelfLoop(object):
    def __init__(self, fill_method="mean") -> None:
        self._fill_method = fill_method

    def __call__(self, data):
        num_nodes = maybe_num_nodes(data.edge_index, None)

        loop_index = torch.arange(0, num_nodes, dtype=torch.long,
                                  device=data.edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)

        if data.edge_attr is not None:
            if self._fill_method == "zero":
                loop_edge_attr = torch.zeros((num_nodes, data.edge_attr.size(1)))
            else:
                loop_edge_attr = scatter_mean(data.edge_attr, data.edge_index[0], dim=0)
            data.edge_attr = torch.cat([data.edge_attr, loop_edge_attr], dim=0)

        data.edge_index = torch.cat([data.edge_index, loop_index], dim=1)

        return data


class ToUndirected(object):
    def __call__(self, data):
        row, col = data.edge_index

        reverse_edge_idx = torch.stack([col, row])

        size = row.size(0)
        direct_label = torch.zeros((2 * size, 2))
        direct_label[:size, 0] = 1
        direct_label[size:, 1] = 1

        data.edge_index = torch.cat([data.edge_index, reverse_edge_idx], dim=-1)
        if data.edge_attr is not None:
            data.edge_attr = torch.cat([torch.cat([data.edge_attr, data.edge_attr], dim=0), direct_label], dim=-1)
        else:
            data.edge_attr = direct_label

        return data


class Split(object):
    def __init__(self, train_pct, seed=None) -> None:
        self._train_pct = train_pct
        self._seed = seed

    def __call__(self, data):
        size = data.x.size(0)
        train_size = int(self._train_pct * size + 1)
        val_size = int(0.5 * size)
        test_size = size - train_size - val_size

        idx = np.array(range(size))
        np.random.seed(self._seed)
        np.random.shuffle(idx)

        train_idx = idx[: train_size]
        val_idx = idx[train_size: train_size + val_size]
        test_idx = idx[train_size + val_size:]

        train_mask, val_mask, test_mask = torch.zeros((size,), dtype=torch.bool), torch.zeros((size,), dtype=torch.bool), torch.zeros((size,), dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        assert((train_mask | val_mask | test_mask).sum() == size)
        assert((train_mask & val_mask).sum() == 0)
        assert((train_mask & test_mask).sum() == 0)

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data


class HandpickEdgeFeature(object):
    def __call__(self, data):
        e, e_idx = data.edge_attr, data.edge_index
        e_mean = scatter_mean(e, e_idx[0], dim=0)
        e_std = scatter_std(e, e_idx[0], dim=0)
        e_min, _ = scatter_min(e, e_idx[0], dim=0)
        e_max, _ = scatter_max(e, e_idx[0], dim=0)

        data.x = torch.cat([data.x, e_mean, e_min, e_max, e_std], dim=-1)

        return data
