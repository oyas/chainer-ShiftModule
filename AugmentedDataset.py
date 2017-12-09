# coding: utf-8

import numpy as np
from chainer import dataset


class AugmentedDataset(dataset.DatasetMixin):

    def __init__(self, data=None):
        self.data = data

    def __len__(self):
        return len(self.data)

    def get_example(self, i):
        x, t = self.data[i]
        if np.random.rand() < 0.8:
            # add white noize
            s = x.shape
            noize = np.random.randn(s[1], s[2]).astype(np.float32)
            x = x + noize * 0.05
            x[x < 0] = 0.
            x[x > 1] = 1.
        if np.random.rand() < 0.5:
            # flip
            x = x.transpose(1, 2, 0)
            x = np.fliplr(x)
            x = x.transpose(2, 0, 1)
        return x, t
