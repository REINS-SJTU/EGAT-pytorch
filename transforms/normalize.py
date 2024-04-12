class NormalizeFeatures(object):
    def __call__(self, data):
        x_mean = data.x.mean(0, keepdim=True)
        x_std = data.x.std(0, keepdim=True)
        data.x = (data.x - x_mean) / x_std
        if data.edge_attr is not None:
            e_mean = data.edge_attr.mean(0, keepdim=True)
            e_std = data.edge_attr.std(0, keepdim=True)
            data.edge_attr = (data.edge_attr - e_mean) / e_std
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class RowNormalizeFeatures(object):
    r"""Row-normalizes node features to sum-up to one."""

    def __call__(self, data):
        data.x = data.x / data.x.sum(1, keepdim=True).clamp(min=1)
        if data.edge_attr is not None:
            data.edge_attr = data.edge_attr / data.edge_attr.sum(1, keepdim=True).clamp(min=1)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
