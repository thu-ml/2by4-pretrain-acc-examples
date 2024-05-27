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


class SparseLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, decay=0.0002, **kwargs):
        super(SparseLinear, self).__init__(in_features, out_features, bias=bias, **kwargs)
        self.N = 2
        self.M = 4

    def forward(self, x):
        x = sparse_linear.apply(x.half(), self.weight.half(), self.bias)
        return x


class sparse_linear(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        weight_sparse = to_sparse_semi_structured(weight)
        ctx.save_for_backward(input, weight_sparse.to_dense(), bias)
        ctx.shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = torch.mm(input, weight_sparse.t())
        if bias is None:
            return output.view(*ctx.shape[:-1], -1)
        else:
            # weight_sparse = to_sparse_semi_structured(weight)
            # ctx.save_for_backward(input, weight_sparse.to_dense(), bias)
            # return F.linear(input, weight_sparse, bias)
            return output.view(*ctx.shape[:-1], -1) + bias

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.half()
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            if grad_output.stride() == (0, 0, 0):
                grad_output = torch.ones_like(grad_output, device=grad_output.device, dtype=grad_output.dtype)
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_input = torch.mm(grad_output, to_sparse_semi_structured(weight.t(), MVUE12=True).t()).view(
                ctx.shape)
        if ctx.needs_input_grad[1]:
            input = input.view(-1, input.shape[-1])
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = torch.mm(to_sparse_semi_structured(grad_output.t(), MVUE12=True), input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class SparseLinearV2(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, **kwargs):
        super(SparseLinearV2, self).__init__(in_features, out_features, bias=bias, **kwargs)

    def forward(self, x):
        x = sparse_linearV2.apply(x.half(), self.weight.half(), self.bias)
        return x


class sparse_linearV2(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        # if bias is None:
        compressed_tensor = torch.empty(
            (weight.numel(),),
            dtype=weight.dtype,
            device=weight.device,
        )
        m, k = weight.shape
        indices_dtype = torch.int16
        sparse, meta = compressed_tensor[: m * k // 2].view(m, -1), compressed_tensor[m * k // 2:].view(
            indices_dtype).view(m, -1)
        _sparse_semi_structured_from_dense_triton(weight, sparse, meta)

        # values
        num_kept_elements = m * k // 2
        values = compressed_tensor[:num_kept_elements].view(m, k // 2)

        # indices
        metadata = compressed_tensor[num_kept_elements:].view(m, -1)
        indices = metadata.view(indices_dtype)

        input_shape = input.shape
        input = input.view(-1, input_shape[-1])

        output = torch._sparse_semi_structured_linear(
            input, values, indices
        )

        ctx.save_for_backward(input,
                              _sparse_semi_structured_to_dense_triton(
                                  compressed_tensor[: m * k // 2].view(m, -1),
                                  compressed_tensor[m * k // 2:].view(indices_dtype).view(m, -1)
                              )
                              , bias)
        if bias is None:
            return output.view(*input_shape[:-1], -1)
        else:
            return output.view(*input_shape[:-1], -1) + bias

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.half()
        input, weight, bias = ctx.saved_tensors
        # if bias is None:
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            if grad_output.stride() == (0, 0, 0):
                grad_output = torch.ones_like(grad_output, device=grad_output.device, dtype=grad_output.dtype)
            # grad_output = grad_output.contiguous()
            weight_T = weight.t()

            compressed_tensor = torch.empty(
                (weight_T.numel(),),
                dtype=weight_T.dtype,
                device=weight_T.device,
            )
            m, k = weight_T.shape
            indices_dtype = torch.int16
            sparse, meta = compressed_tensor[: m * k // 2].view(m, -1), compressed_tensor[m * k // 2:].view(
                indices_dtype).view(m, -1)
            _sparse_semi_structured_from_dense_triton_MVUE12(weight_T, sparse, meta)

            # values
            num_kept_elements = m * k // 2
            values = compressed_tensor[:num_kept_elements].view(m, k // 2)

            # indices
            metadata = compressed_tensor[num_kept_elements:].view(m, -1)
            indices = metadata.view(indices_dtype)

            grad_output_shape = grad_output.shape
            grad_output = grad_output.view(-1, grad_output_shape[-1])

            grad_input = torch._sparse_semi_structured_linear(
                grad_output, values, indices
            )

            grad_input = grad_input.view(*grad_output_shape[:-1], -1)
            grad_output = grad_output.view(grad_output_shape)
        if ctx.needs_input_grad[1]:
            grad_output_T = grad_output.view(-1, grad_output.shape[-1]).t()
            input_shape = input.shape
            input = input.view(-1, input.shape[-1])

            compressed_tensor = torch.empty(
                (grad_output_T.numel(),),
                dtype=grad_output_T.dtype,
                device=grad_output_T.device,
            )
            m, k = grad_output_T.shape
            indices_dtype = torch.int16
            sparse, meta = compressed_tensor[: m * k // 2].view(m, -1), compressed_tensor[m * k // 2:].view(
                indices_dtype).view(m, -1)
            _sparse_semi_structured_from_dense_triton_MVUE12(grad_output_T, sparse, meta)
            # values
            num_kept_elements = m * k // 2
            values = compressed_tensor[:num_kept_elements].view(m, k // 2)

            # indices
            metadata = compressed_tensor[num_kept_elements:].view(m, -1)
            indices = metadata.view(indices_dtype)

            transposed_grad_weight = torch._sparse_semi_structured_linear(
                input.t(), values, indices
            )
            grad_weight = transposed_grad_weight.t()

            input = input.view(input_shape)
        if ctx.needs_input_grad[2]:
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_bias = torch.sum(grad_output.view(-1, grad_output.shape[-1]), dim=0)
        return grad_input, grad_weight, grad_bias


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
        if self.weight.mode == 'dense':
            x = F.linear(x, self.weight, self.bias)
        else:
            self.weight.mask = self.weight.mask.to(device=self.weight.device)
            if self.weight.counter % self.weight.freq == 0 and self.weight.cnt == 0:
                _, self.weight.mask = self.transposable_sparse(self.weight)
            if self.weight.cnt == 0:
                self.weight.weight_sparse = to_sparse_semi_structured(self.weight, mask=self.weight.mask,
                                                                      dtype=torch.float16)
                self.weight.weight_sparse_T = to_sparse_semi_structured(self.weight.T, mask=self.weight.mask.T,
                                                                        dtype=torch.float16)
            with autocast(device_type='cuda', dtype=torch.float16):
                x = sparse_linear_transpose.apply(x, self.weight, self.weight.weight_sparse,
                                                  self.weight.weight_sparse_T,
                                                  self.bias)

        if self.training:
            if self.weight.cnt == 0:
                self.weight.counter += 1
            self.weight.step += 1
            self.weight.cnt += 1
        return x


class sparse_linear_transpose(autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx, input, weight, weight_sparse, weight_sparse_T, bias):
        ctx.save_for_backward(input, weight_sparse_T, bias)
        ctx.shape = input.shape
        input = input.view(-1, input.shape[-1])
        output = torch.mm(input, weight_sparse.t())
        if bias is None:
            return output.view(*ctx.shape[:-1], -1)
        else:
            return output.view(*ctx.shape[:-1], -1) + bias

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output
        input, weight_T, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            if grad_output.stride() == (0, 0, 0):
                grad_output = torch.ones_like(grad_output, device=grad_output.device, dtype=grad_output.dtype)
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_input = torch.mm(grad_output, weight_T.t()).view(
                ctx.shape)
        if ctx.needs_input_grad[1]:
            input = input.view(-1, input.shape[-1])
            grad_output = grad_output.view(-1, grad_output.shape[-1])
            grad_weight = torch.mm(to_sparse_semi_structured(grad_output.t(), MVUE24=True), input)
        if ctx.needs_input_grad[4]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, None, None, grad_bias
