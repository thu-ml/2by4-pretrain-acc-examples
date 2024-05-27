import torch
from torch import autograd, nn, autocast
import torch.nn.functional as F

from itertools import repeat

from torch.cuda.amp import custom_fwd, custom_bwd

from sparse._semi_structured_conversions import _sparse_semi_structured_to_dense_triton, \
    _sparse_semi_structured_from_dense_triton, _sparse_semi_structured_from_dense_triton_MVUE12
from sparse.semi_structured import to_sparse_semi_structured, SparseSemiStructuredTensor
from sparse.transposable_semi_structured import TransposableSparse

# from torch._six import container_abcs
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


class SparseLinearTranspose(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, func=lambda step: 'dense',
                 **kwargs):
        super(SparseLinearTranspose, self).__init__(in_features, out_features, bias=bias, **kwargs)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)
