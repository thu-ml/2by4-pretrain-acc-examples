import torch
from torch import nn, autograd
import torch.nn.functional as F

from sparse import soft_threshold24_triton


class SoftThreshold(autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale):
        weight_temp = weight.detach()
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        return weight_sparse * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class FP8SparseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super(FP8SparseLinear, self).__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.register_buffer('scale', torch.tensor(0.))

    def get_sparse_weights(self):
        return SoftThreshold.apply(self.weight, self.scale)

    @torch.no_grad()
    def init_scale(self):
        weight = self.weight.cuda()
        weight_temp = weight.detach()
        weight_sparse, _ = soft_threshold24_triton(weight_temp)
        weight.scale = torch.dot(torch.flatten(weight), torch.flatten(weight_sparse)) / torch.dot(
            torch.flatten(weight_sparse), torch.flatten(weight_sparse))
        self.scale.copy_(weight.scale.cpu())
        self.weight.scale = self.scale

    def forward(self, x):
        w = self.get_sparse_weights()
        x = F.linear(x, w, self.bias)
        return x
