import torch
from torch import autograd, nn
import torch.nn.functional as F

from itertools import repeat

from torch.cuda.amp import custom_fwd, custom_bwd
from torch.nn import Parameter

from legacy import transposable_sparse_mask, MVUE24_approx_triton

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


class SparseTranspose(autograd.Function):
    """" Prune the unimprotant weight for the forwards phase but pass the gradient to dense weight using SR-STE in the backwards phase"""

    @staticmethod
    def forward(ctx, weight, N, M, counter, freq, absorb_mean):
        weight.mask = weight.mask.to(weight.device)
        weight.old_mask = weight.old_mask.to(weight.device)

        output = weight.clone()

        ctx.flipped = False

        if counter % freq == 0 and weight.cnt == 0:
            weight_temp = weight.detach().abs()
            weight_mask = transposable_sparse_mask(weight_temp, abs=True).bool()

            weight.old_mask.data.copy_(weight.mask.data)
            # assert not (weight.old_mask+weight_mask).equal(torch.ones_like(weight_mask))

            weight.mask = weight_mask

            # ctx.save_for_backward(2 * (weight.mask - weight_mask))
            ctx.flipped = False

        weight.cnt += 1
        return output * weight.mask, weight.mask
        # return output , weight.mask

    @staticmethod
    def backward(ctx, grad_output, _):
        return grad_output, None, None, None, None, None, None


class SparseLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True, N=2, M=4, decay=0.0002, **kwargs):
        self.N = N
        self.M = M
        super(SparseLinear, self).__init__(in_features, out_features, bias=bias)
        self.weight.counter = 0
        self.weight.freq = 10
        self.weight.cnt = 0

        self.weight.mask = torch.ones_like(self.weight).bool()
        self.weight.old_mask = torch.ones_like(self.weight).bool()

        self.weight.mode = 'sparse'

    def get_sparse_weights(self):
        return SparseTranspose.apply(self.weight, self.N, self.M, self.weight.counter, self.weight.freq, False)

    def forward(self, x):
        if self.weight.cnt == 0:
            self.weight.counter += 1
        self.weight.freq = 10
        w, mask = self.get_sparse_weights()
        # setattr(self.weight, "mask", mask)
        if self.weight.mode == 'sparse':
            x = sparse_linear.apply(x, w, self.bias)
        else:
            x = F.linear(x, self.weight, self.bias)
        return x


class sparse_linear(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight.t(), bias)
        ctx.shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = torch.mm(input, weight.t())
        if bias is None:
            return output.view(*ctx.shape[:-1], -1)
        else:
            return output.view(*ctx.shape[:-1], -1) + bias

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight_T, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_input = torch.mm(grad_output, weight_T.t()).view(ctx.shape)
        if ctx.needs_input_grad[1]:
            input = input.view(-1, input.shape[-1])
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = torch.mm(MVUE24_approx_triton(grad_output.t()), input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias
