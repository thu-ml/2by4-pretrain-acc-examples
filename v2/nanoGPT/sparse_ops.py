import torch
from torch import nn, autograd
from torch.cuda.amp import custom_fwd, custom_bwd

from sparse import matmul, MVUE24_approx_triton, soft_threshold24_triton


def fake_fp8_mm(a, b, dtype):
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    margin = 0
    FP8_MAX = maximum_representable_value(dtype)
    amax_a = a.abs().max()
    amax_b = b.abs().max()
    new_scaling_factor_a = (FP8_MAX / amax_a) / (2 ** margin)
    new_scaling_factor_b = (FP8_MAX / amax_b) / (2 ** margin)
    a = (a * new_scaling_factor_a).to(dtype).contiguous()
    b = (b * new_scaling_factor_b).to(dtype).t().contiguous().t()
    output = matmul(a, b, c_dtype=torch.float32) / (new_scaling_factor_a * new_scaling_factor_b)
    return output


def maximum_representable_value(dtype):
    if dtype == torch.float8_e5m2:
        return 57344.
    elif dtype == torch.float8_e4m3fn:
        return 448.
    else:
        raise ValueError("Unsupported dtype")


class fp8_linear(autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        ctx.shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = fake_fp8_mm(input, weight.t(), torch.float8_e4m3fn)
        if bias is None:
            return output.view(*ctx.shape[:-1], -1)
        else:
            return output.view(*ctx.shape[:-1], -1) + bias

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output.half()
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            if grad_output.stride() == (0, 0, 0):
                grad_output = torch.ones_like(grad_output, device=grad_output.device, dtype=grad_output.dtype)
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_input = fake_fp8_mm(grad_output, weight, torch.float8_e5m2).view(ctx.shape)
        if ctx.needs_input_grad[1]:
            input = input.view(-1, input.shape[-1])
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = fake_fp8_mm(MVUE24_approx_triton(grad_output.t()), input, torch.float8_e5m2)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


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
        x = fp8_linear.apply(x, w, self.bias)
        return x
