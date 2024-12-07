import torch
from torch import nn
import torch.nn.functional as F

from sparse import TransposableSparse, SparseSemiStructuredTensor


class SparseLinearTranspose(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, func=lambda step: 'dense',
                 **kwargs):
        super(SparseLinearTranspose, self).__init__(in_features, out_features, bias=bias, **kwargs)
        self.weight.freq = 40  # update freq

        self.weight.cnt = 0  # how many steps after an optim step
        self.weight.counter = 0  # how many optim steps
        self.weight.step = 0  # total training step

        self.weight.mask = torch.ones_like(self.weight, dtype=torch.bool)
        self.weight.weight_sparse = None
        self.weight.weight_sparse_T = None
        self.weight.mode = 'sparse'
        self.func = func

        self.transposable_sparse = TransposableSparse(abs=True)
        SparseSemiStructuredTensor._FORCE_CUTLASS = True  # we won't need this later

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
