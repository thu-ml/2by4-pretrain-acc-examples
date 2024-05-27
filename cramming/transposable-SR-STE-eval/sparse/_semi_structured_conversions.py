from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


def _sparse_semi_structured_from_dense(dense):
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")

    if m % 32 != 0:
        raise RuntimeError(
            f"Number rows columns of dense matrix {m} must be divisible by 32"
        )
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}"
        )
    meta_ncols = k // (4 * quadbits_per_meta_elem)

    dense_4 = dense.view(-1, k // 4, 4)
    # m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)
    A, B, C, D = dense_4.abs().unbind(-1)
    x1, x2, x3, x4, x5, x6 = A > B, A > C, A > D, B > C, B > D, C > D
    m0, m1, m2, m3 = x2 & x3 | x1 & x2 | x1 & x3, ~x1 & x5 | x4 & x5 | ~x1 & x4, ~x2 & ~x4 | ~x2 & x6 | ~x4 & x6, ~x3 & ~x5 | ~x3 & ~x6 | ~x5 & ~x6

    # Encoding quadruples of True/False values as follows:
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # Thus, lower two bits in the encoding are index of the True value
    # at the lowest index in the quadruple, and the higher two bits in
    # the encoding are index of the other True value in the quadruple.
    # In case there are less than two True values, than False value or
    # values at some index or indices are considered True for the
    # encoding.  In case there are more than two True values, then the
    # excess True value(s) at some indices are considered False for
    # the encoding.  The exact encodings used for these cases are as
    # follows:
    #     [False, False, False, False] -> 0b1110
    #     [False, False, False, True ] -> 0b1110
    #     [False, False, True,  False] -> 0b1110
    #     [False, True,  False, False] -> 0b1101
    #     [False, True,  True,  True ] -> 0b1001
    #     [True,  False, False, False] -> 0b1100
    #     [True,  False, True,  True ] -> 0b1000
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b1000
    #     [True,  True,  True,  True ] -> 0b1000
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.

    bit0 = ~m0 & m1
    bit1 = ~m0 & ~m1
    bit2 = bit1 | ~m2
    bit3 = bit0 | ~m1 | m2
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))
    sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
    sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    if quadbits_per_meta_elem == 4:
        meta = (
                meta_n[:, :, 0]
                | (meta_n[:, :, 1] << 4)
                | (meta_n[:, :, 2] << 8)
                | (meta_n[:, :, 3] << 12)
        )
    elif quadbits_per_meta_elem == 8:
        meta = (
                meta_n[:, :, 0]
                | (meta_n[:, :, 1] << 4)
                | (meta_n[:, :, 2] << 8)
                | (meta_n[:, :, 3] << 12)
                | (meta_n[:, :, 4] << 16)
                | (meta_n[:, :, 5] << 20)
                | (meta_n[:, :, 6] << 24)
                | (meta_n[:, :, 7] << 28)
        )

    # Metadata values are now to be reshuffled in a way given in
    # reorder_meta() function, in
    # tools/util/include/cutlass/util/host_reorder.h file of CUTLASS
    # source tree.  Furthermore, CUTLASS template for sparse GEMM
    # decides upon layout of this matrix, and at the moment for the
    # sparse GEMM executed on tensor cores, this is layout described
    # by ColumnMajorInterleaved<2> data structure, in
    # include/cutlass/layout/matrix.h of CUTLASS source tree.  The
    # reordering of meta matrix into meta_reordered matrix calculated
    # according to these segments of CUTLASS code is given below.
    # However, this calculation produces offsets for scatter access
    # from metadata matrix to redordered metadata matrix, and gather
    # pattern is more efficient.  For this reason, the scatter offsets
    # are reverted and printed, through enabling commented block at
    # the end of following code.  Resulting gather offsets are then
    # analyzed, on several (m, k) value pairs (in particular: (32,
    # 128), (32, 256), (64, 128) and (64, 256)), and the code that
    # follows this comment is written to reproduce these gather offsets.
    #
    #    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    #    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)
    #
    #    # Reorder the rows, then swizzle the 2x2 blocks.
    #    group = 32 if meta_dtype.itemsize == 2 else 16
    #    interweave = 4 if meta_dtype.itemsize == 2 else 2
    #    dst_rows = (
    #        dst_rows // group * group
    #        + (dst_rows % 8) * interweave
    #        + (dst_rows % group) // 8
    #    )
    #
    #    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    #    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    #    dst_rows += topright - bottomleft
    #    dst_cols -= topright - bottomleft
    #
    #    # Assumed that meta tensor is to be stored in CUTLASS
    #    # InterleavedColumnMajor layout, and reverse engineered
    #    # corresponding code to store values into this tensor.
    #    interleave = 2
    #    cols_maj = dst_cols // interleave
    #    cols_min = dst_cols % interleave
    #    meta_reordered_offsets = (
    #        cols_maj * m * interleave + dst_rows * interleave + cols_min
    #    )
    #
    #    meta_reordered = torch.empty((m, meta_ncols), dtype=meta_dtype, device=device)
    #    meta_reordered.view(-1)[meta_reordered_offsets.view(-1)] = meta.view(-1)
    #
    #    # Uncomment to have gather pattern for meta_reordered printed
    #    #
    #    #offsets = torch.empty(
    #    #    (m, meta_ncols), dtype=meta_reordered_offsets.dtype, device=device
    #    #)
    #    #offsets.view(-1)[meta_reordered_offsets.view(-1)] = torch.arange(
    #    #    0, m * meta_ncols, dtype=meta_reordered_offsets.dtype, device=device
    #    #)
    #    #torch.set_printoptions(threshold=1000000)
    #    #print("------------------------------------------------------------")
    #    #print("dtype =", dtype, ", m =", m, ", k =", k, ", meta_ncols =", meta_ncols)
    #    #print(offsets.view(-1))
    #

    # No point to try to understand this code: as mentioned in the
    # comment above it is written to reproduce gather offsets, as
    # these would be calculated by CUTLASS, and to be efficient, but
    # it contains several magic values and magic calculations that
    # make it rather hard to read, let alone understand.
    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = 32
        magic2 = 16
        magic3 = k // 2
        magic4 = [0, k // 4, 1, k // 4 + 1]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = 64
        magic2 = 32
        magic3 = 2 * k
        magic4 = [0, k // 2, 1, k // 2 + 1, k, 3 * k // 2, k + 1, 3 * k // 2 + 1]
    tmp0 = torch.zeros(m * meta_ncols, dtype=torch.int64, device=device)
    tmp1 = (
            tmp0.view(meta_ncols // 2, -1)
            + torch.arange(0, meta_ncols, 2, device=device).view(meta_ncols // 2, 1)
    ).view(-1, magic1)
    tmp2 = (
        (
                torch.arange(0, 8, device=device).view(-1, 1)
                * torch.ones((magic0,), dtype=torch.int64, device=device)
                * meta_ncols
        )
        .view(-1)
        .repeat(m * meta_ncols // magic1)
        .view(-1, magic1)
    )
    tmp3 = (torch.arange(0, m // magic2, device=device).view(-1, 1) * magic3).repeat(
        meta_ncols // 2, magic1
    )
    tmp4 = torch.tensor(magic4, device=device).repeat(tmp3.shape[0], 8)
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4

    meta_reordered = torch.gather(meta.view(-1), 0, meta_offsets.view(-1)).view(
        m, meta_ncols
    )
    return (sparse, meta_reordered)


def _sparse_semi_structured_to_dense(sparse, meta_reordered):
    if sparse.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor"
        )

    m, k = sparse.shape
    device = sparse.device

    if meta_reordered.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor"
        )
    if meta_reordered.device != device:
        raise RuntimeError(
            f"Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device"
        )

    meta_dtype = meta_reordered.dtype
    if meta_dtype not in (torch.int16, torch.int32):
        raise RuntimeError(f"Invalid datatype {meta_dtype} of meta matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4

    meta_nrows, meta_ncols = meta_reordered.shape
    if meta_nrows != m:
        raise RuntimeError(
            f"Number of rows of meta matrix {meta_nrows} must be equal to number of columns of spase matrix {m}"
        )
    if meta_ncols * 4 * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(
            f"Number of columns of sparse matrix {k} different from the {meta_ncols * 4 * quadbits_per_meta_elem // 2}, "
            "expected according to the number of columns of meta matrix"
        )

    if meta_dtype == torch.int32:
        magic0 = 4
        magic1 = [0, 1, 32, 33]
    elif meta_dtype == torch.int16:
        magic0 = 8
        magic1 = [0, 1, 4, 5]
    tmp1 = torch.tensor([0, 2], dtype=torch.int64, device=device).repeat(
        meta_nrows, meta_ncols // 2
    )
    tmp2 = (
        (torch.arange(0, meta_ncols // 2, device=device) * 2 * meta_nrows)
        .view(-1, 1)
        .repeat(1, 2)
        .view(-1)
        .repeat(m, 1)
    )
    tmp3 = (
        (torch.arange(0, 8, device=device) * magic0)
        .view(-1, 1)
        .repeat(m // 8, meta_ncols)
    )
    tmp4 = (
        torch.tensor(magic1, device=device)
        .view(-1, 1)
        .repeat(1, 8 * meta_ncols)
        .repeat(meta_nrows // 32, 1)
        .view(meta_nrows, meta_ncols)
    )
    tmp5 = (
        (torch.arange(0, meta_nrows // 32, device=device) * 64)
        .view(-1, 1)
        .repeat(1, 32 * meta_ncols)
        .view(meta_nrows, meta_ncols)
    )
    meta_offsets = tmp1 + tmp2 + tmp3 + tmp4 + tmp5

    meta = torch.gather(meta_reordered.view(-1), 0, meta_offsets.view(-1)).view(
        m, meta_ncols
    )

    meta_2 = torch.empty(
        (m, meta_ncols, 2 * quadbits_per_meta_elem), dtype=meta_dtype, device=device
    )
    if quadbits_per_meta_elem == 4:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
    elif quadbits_per_meta_elem == 8:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
        meta_2[:, :, 8] = (meta >> 16) & 0b11
        meta_2[:, :, 9] = (meta >> 18) & 0b11
        meta_2[:, :, 10] = (meta >> 20) & 0b11
        meta_2[:, :, 11] = (meta >> 22) & 0b11
        meta_2[:, :, 12] = (meta >> 24) & 0b11
        meta_2[:, :, 13] = (meta >> 26) & 0b11
        meta_2[:, :, 14] = (meta >> 28) & 0b11
        meta_2[:, :, 15] = (meta >> 30) & 0b11

    dense_offsets = meta_2.view(-1) + (
            torch.arange(0, m * k // 2, device=device) * 4
    ).view(-1, 1).repeat(1, 2).view(-1)

    dense = torch.zeros((m * 2 * k,), dtype=sparse.dtype, device=device)
    dense.scatter_(0, dense_offsets, sparse.view(-1))

    return dense.view(m, 2 * k)


def sparse_semi_structured_from_dense(dense):
    from torch._dynamo.utils import is_compile_supported
    if is_compile_supported(dense.device.type):
        kernel = torch.compile(_sparse_semi_structured_from_dense)
        return kernel(dense)

    return _sparse_semi_structured_from_dense(dense)


def sparse_semi_structured_to_dense(sparse, meta_reordered):
    from torch._dynamo.utils import is_compile_supported
    if is_compile_supported(sparse.device.type):
        kernel = torch.compile(_sparse_semi_structured_to_dense)
        return kernel(sparse, meta_reordered)

    return _sparse_semi_structured_to_dense(sparse, meta_reordered)


@triton.jit
def _sparse_semi_structured_from_dense_kernel(
        dense_ptr,
        sparse_ptr,
        meta_reordered_ptr,
        dense_row_stride,
        sparse_row_stride,
        dense_col_stride,
        m, k,  # dense.shape
        BLOCK_SIZE: tl.constexpr,

):
    row_idx = tl.program_id(0)

    col_idx = 16 * tl.arange(0, BLOCK_SIZE // 16)
    m00 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))  # A0
    col_idx += 1
    m10 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))  # B0
    col_idx += 1
    m20 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m30 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))

    col_idx += 1
    m01 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))  # A1
    col_idx += 1
    m11 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m21 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m31 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))

    col_idx += 1
    m02 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m12 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m22 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m32 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))

    col_idx += 1
    m03 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m13 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m23 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m33 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))

    x10, x20, x30, x40, x50, x60 = tl.abs(m00) > tl.abs(m10), tl.abs(m00) > tl.abs(m20), tl.abs(m00) > tl.abs(
        m30), tl.abs(m10) > tl.abs(m20), tl.abs(m10) > tl.abs(m30), tl.abs(m20) > tl.abs(m30)
    m00_, m10_, m20_, m30_ = x20 & x30 | x10 & x20 | x10 & x30, ~x10 & x50 | x40 & x50 | ~x10 & x40, ~x20 & ~x40 | ~x20 & x60 | ~x40 & x60, ~x30 & ~x50 | ~x30 & ~x60 | ~x50 & ~x60

    x11, x21, x31, x41, x51, x61 = tl.abs(m01) > tl.abs(m11), tl.abs(m01) > tl.abs(m21), tl.abs(m01) > tl.abs(
        m31), tl.abs(m11) > tl.abs(m21), tl.abs(m11) > tl.abs(m31), tl.abs(m21) > tl.abs(m31)
    m01_, m11_, m21_, m31_ = x21 & x31 | x11 & x21 | x11 & x31, ~x11 & x51 | x41 & x51 | ~x11 & x41, ~x21 & ~x41 | ~x21 & \
                             x61 | ~x41 & x61, ~x31 & ~x51 | ~x31 & ~x61 | ~x51 & ~x61

    x12, x22, x32, x42, x52, x62 = tl.abs(m02) > tl.abs(m12), tl.abs(m02) > tl.abs(m22), tl.abs(m02) > tl.abs(
        m32), tl.abs(m12) > tl.abs(m22), tl.abs(m12) > tl.abs(m32), tl.abs(m22) > tl.abs(m32)
    m02_, m12_, m22_, m32_ = x22 & x32 | x12 & x22 | x12 & x32, ~x12 & x52 | x42 & x52 | ~x12 & x42, ~x22 & ~x42 | ~x22 & \
                             x62 | ~x42 & x62, ~x32 & ~x52 | ~x32 & ~x62 | ~x52 & ~x62

    x13, x23, x33, x43, x53, x63 = tl.abs(m03) > tl.abs(m13), tl.abs(m03) > tl.abs(m23), tl.abs(m03) > tl.abs(
        m33), tl.abs(m13) > tl.abs(m23), tl.abs(m13) > tl.abs(m33), tl.abs(m23) > tl.abs(m33)
    m03_, m13_, m23_, m33_ = x23 & x33 | x13 & x23 | x13 & x33, ~x13 & x53 | x43 & x53 | ~x13 & x43, ~x23 & ~x43 | ~x23 & \
                             x63 | ~x43 & x63, ~x33 & ~x53 | ~x33 & ~x63 | ~x53 & ~x63

    # initial codes are as bellow.
    # m00_, m10_, m20_, m30_ = m00.to(tl.int1), m10.to(tl.int1), m20.to(tl.int1), m30.to(tl.int1)
    # m01_, m11_, m21_, m31_ = m01.to(tl.int1), m11.to(tl.int1), m21.to(tl.int1), m31.to(tl.int1)
    # m02_, m12_, m22_, m32_ = m02.to(tl.int1), m12.to(tl.int1), m22.to(tl.int1), m32.to(tl.int1)
    # m03_, m13_, m23_, m33_ = m03.to(tl.int1), m13.to(tl.int1), m23.to(tl.int1), m33.to(tl.int1)

    bit00 = ~m00_ & m10_
    bit10 = ~m00_ & ~m10_
    bit20 = bit10 | ~m20_
    bit30 = bit00 | ~m10_ | m20_
    idxs00 = bit00 | (bit10.to(tl.int64) << 1)
    idxs10 = bit20 | (bit30.to(tl.int64) << 1)
    sparse00 = tl.where(bit10, tl.where(bit00, m30, m20), tl.where(bit00, m10, m00))
    sparse10 = tl.where(bit30, tl.where(bit20, m30, m20), tl.where(bit20, m10, m00))

    bit01 = ~m01_ & m11_
    bit11 = ~m01_ & ~m11_
    bit21 = bit11 | ~m21_
    bit31 = bit01 | ~m11_ | m21_
    idxs01 = bit01 | (bit11.to(tl.int64) << 1)
    idxs11 = bit21 | (bit31.to(tl.int64) << 1)
    sparse01 = tl.where(bit11, tl.where(bit01, m31, m21), tl.where(bit01, m11, m01))
    sparse11 = tl.where(bit31, tl.where(bit21, m31, m21), tl.where(bit21, m11, m01))

    bit02 = ~m02_ & m12_
    bit12 = ~m02_ & ~m12_
    bit22 = bit12 | ~m22_
    bit32 = bit02 | ~m12_ | m22_
    idxs02 = bit02 | (bit12.to(tl.int64) << 1)
    idxs12 = bit22 | (bit32.to(tl.int64) << 1)
    sparse02 = tl.where(bit12, tl.where(bit02, m32, m22), tl.where(bit02, m12, m02))
    sparse12 = tl.where(bit32, tl.where(bit22, m32, m22), tl.where(bit22, m12, m02))

    bit03 = ~m03_ & m13_
    bit13 = ~m03_ & ~m13_
    bit23 = bit13 | ~m23_
    bit33 = bit03 | ~m13_ | m23_
    idxs03 = bit03 | (bit13.to(tl.int64) << 1)
    idxs13 = bit23 | (bit33.to(tl.int64) << 1)
    sparse03 = tl.where(bit13, tl.where(bit03, m33, m23), tl.where(bit03, m13, m03))
    sparse13 = tl.where(bit33, tl.where(bit23, m33, m23), tl.where(bit23, m13, m03))

    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16), sparse00,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 1, sparse10,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 1 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 2, sparse01,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 2 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 3, sparse11,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 3 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 4, sparse02,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 4 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 5, sparse12,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 5 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 6, sparse03,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 6 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 7, sparse13,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 7 < k / 2)

    meta_40 = idxs00 | (idxs10 << 2)
    meta_41 = idxs01 | (idxs11 << 2)
    meta_42 = idxs02 | (idxs12 << 2)
    meta_43 = idxs03 | (idxs13 << 2)
    meta = (
            meta_40
            | (meta_41 << 4)
            | (meta_42 << 8)
            | (meta_43 << 12)
    )

    group, interweave = 32, 4

    dest_row = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8
    if dest_row % 2 == 0:
        dest_row_ = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8 + tl.arange(0, BLOCK_SIZE // 16) % 2
        dest_col_ = tl.arange(0, BLOCK_SIZE // 16) // 2 * 2
        index = (dest_col_ // 2) * m * 2 + dest_row_ * 2 + dest_col_ % 2
        tl.store(meta_reordered_ptr + index, meta, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    else:
        dest_row_ = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8 - (
                tl.arange(0, BLOCK_SIZE // 16) + 1) % 2
        dest_col_ = tl.arange(0, BLOCK_SIZE // 16) // 2 * 2 + 1
        index = (dest_col_ // 2) * m * 2 + dest_row_ * 2 + dest_col_ % 2
        tl.store(meta_reordered_ptr + index, meta, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)


@triton.jit
def _sparse_semi_structured_from_dense_with_mask_kernel(
        dense_ptr,
        sparse_ptr,
        meta_reordered_ptr,
        mask_ptr,
        dense_row_stride,
        sparse_row_stride,
        mask_row_stride,
        dense_col_stride,
        mask_col_stride,
        m, k,  # dense.shape
        BLOCK_SIZE: tl.constexpr,

):
    row_idx = tl.program_id(0)

    col_idx = 16 * tl.arange(0, BLOCK_SIZE // 16)
    m00_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)  # A0
    m00 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m00_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))  # A0
    col_idx += 1
    m10_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)  # B0
    m10 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m10_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))  # B0
    col_idx += 1
    m20_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m20 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m20_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))
    col_idx += 1
    m30_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m30 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m30_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))

    col_idx += 1
    m01_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)  # A1
    m01 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m01_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))  # A1
    col_idx += 1
    m11_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m11 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m11_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))
    col_idx += 1
    m21_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m21 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m21_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))
    col_idx += 1
    m31_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m31 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m31_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))

    col_idx += 1
    m02_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m02 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m02_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))
    col_idx += 1
    m12_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m12 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m12_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))
    col_idx += 1
    m22_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m22 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m22_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))
    col_idx += 1
    m32_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m32 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m32_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))

    col_idx += 1
    m03_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m03 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m03_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))
    col_idx += 1
    m13_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m13 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m13_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))
    col_idx += 1
    m23_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m23 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m23_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))
    col_idx += 1
    m33_ = tl.load(mask_ptr + row_idx * mask_row_stride + col_idx * mask_col_stride,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                   other=-float('inf')).to(tl.int1)
    m33 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=m33_ & (tl.arange(0, BLOCK_SIZE // 16) < k // 16),
                  other=-float('inf'))

    # x10, x20, x30, x40, x50, x60 = tl.abs(m00) > tl.abs(m10), tl.abs(m00) > tl.abs(m20), tl.abs(m00) > tl.abs(
    #     m30), tl.abs(m10) > tl.abs(m20), tl.abs(m10) > tl.abs(m30), tl.abs(m20) > tl.abs(m30)
    # m00_, m10_, m20_, m30_ = x20 & x30 | x10 & x20 | x10 & x30, ~x10 & x50 | x40 & x50 | ~x10 & x40, ~x20 & ~x40 | ~x20 & x60 | ~x40 & x60, ~x30 & ~x50 | ~x30 & ~x60 | ~x50 & ~x60
    # 
    # x11, x21, x31, x41, x51, x61 = tl.abs(m01) > tl.abs(m11), tl.abs(m01) > tl.abs(m21), tl.abs(m01) > tl.abs(
    #     m31), tl.abs(m11) > tl.abs(m21), tl.abs(m11) > tl.abs(m31), tl.abs(m21) > tl.abs(m31)
    # m01_, m11_, m21_, m31_ = x21 & x31 | x11 & x21 | x11 & x31, ~x11 & x51 | x41 & x51 | ~x11 & x41, ~x21 & ~x41 | ~x21 & \
    #                          x61 | ~x41 & x61, ~x31 & ~x51 | ~x31 & ~x61 | ~x51 & ~x61
    # 
    # x12, x22, x32, x42, x52, x62 = tl.abs(m02) > tl.abs(m12), tl.abs(m02) > tl.abs(m22), tl.abs(m02) > tl.abs(
    #     m32), tl.abs(m12) > tl.abs(m22), tl.abs(m12) > tl.abs(m32), tl.abs(m22) > tl.abs(m32)
    # m02_, m12_, m22_, m32_ = x22 & x32 | x12 & x22 | x12 & x32, ~x12 & x52 | x42 & x52 | ~x12 & x42, ~x22 & ~x42 | ~x22 & \
    #                          x62 | ~x42 & x62, ~x32 & ~x52 | ~x32 & ~x62 | ~x52 & ~x62
    # 
    # x13, x23, x33, x43, x53, x63 = tl.abs(m03) > tl.abs(m13), tl.abs(m03) > tl.abs(m23), tl.abs(m03) > tl.abs(
    #     m33), tl.abs(m13) > tl.abs(m23), tl.abs(m13) > tl.abs(m33), tl.abs(m23) > tl.abs(m33)
    # m03_, m13_, m23_, m33_ = x23 & x33 | x13 & x23 | x13 & x33, ~x13 & x53 | x43 & x53 | ~x13 & x43, ~x23 & ~x43 | ~x23 & \
    #                          x63 | ~x43 & x63, ~x33 & ~x53 | ~x33 & ~x63 | ~x53 & ~x63

    # initial codes are as bellow.
    # m00_, m10_, m20_, m30_ = m00.to(tl.int1), m10.to(tl.int1), m20.to(tl.int1), m30.to(tl.int1)
    # m01_, m11_, m21_, m31_ = m01.to(tl.int1), m11.to(tl.int1), m21.to(tl.int1), m31.to(tl.int1)
    # m02_, m12_, m22_, m32_ = m02.to(tl.int1), m12.to(tl.int1), m22.to(tl.int1), m32.to(tl.int1)
    # m03_, m13_, m23_, m33_ = m03.to(tl.int1), m13.to(tl.int1), m23.to(tl.int1), m33.to(tl.int1)

    bit00 = ~m00_ & m10_
    bit10 = ~m00_ & ~m10_
    bit20 = bit10 | ~m20_
    bit30 = bit00 | ~m10_ | m20_
    idxs00 = bit00 | (bit10.to(tl.int64) << 1)
    idxs10 = bit20 | (bit30.to(tl.int64) << 1)
    sparse00 = tl.where(bit10, tl.where(bit00, m30, m20), tl.where(bit00, m10, m00))
    sparse10 = tl.where(bit30, tl.where(bit20, m30, m20), tl.where(bit20, m10, m00))

    bit01 = ~m01_ & m11_
    bit11 = ~m01_ & ~m11_
    bit21 = bit11 | ~m21_
    bit31 = bit01 | ~m11_ | m21_
    idxs01 = bit01 | (bit11.to(tl.int64) << 1)
    idxs11 = bit21 | (bit31.to(tl.int64) << 1)
    sparse01 = tl.where(bit11, tl.where(bit01, m31, m21), tl.where(bit01, m11, m01))
    sparse11 = tl.where(bit31, tl.where(bit21, m31, m21), tl.where(bit21, m11, m01))

    bit02 = ~m02_ & m12_
    bit12 = ~m02_ & ~m12_
    bit22 = bit12 | ~m22_
    bit32 = bit02 | ~m12_ | m22_
    idxs02 = bit02 | (bit12.to(tl.int64) << 1)
    idxs12 = bit22 | (bit32.to(tl.int64) << 1)
    sparse02 = tl.where(bit12, tl.where(bit02, m32, m22), tl.where(bit02, m12, m02))
    sparse12 = tl.where(bit32, tl.where(bit22, m32, m22), tl.where(bit22, m12, m02))

    bit03 = ~m03_ & m13_
    bit13 = ~m03_ & ~m13_
    bit23 = bit13 | ~m23_
    bit33 = bit03 | ~m13_ | m23_
    idxs03 = bit03 | (bit13.to(tl.int64) << 1)
    idxs13 = bit23 | (bit33.to(tl.int64) << 1)
    sparse03 = tl.where(bit13, tl.where(bit03, m33, m23), tl.where(bit03, m13, m03))
    sparse13 = tl.where(bit33, tl.where(bit23, m33, m23), tl.where(bit23, m13, m03))

    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16), sparse00,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 1, sparse10,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 1 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 2, sparse01,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 2 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 3, sparse11,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 3 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 4, sparse02,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 4 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 5, sparse12,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 5 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 6, sparse03,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 6 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 7, sparse13,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 7 < k / 2)

    meta_40 = idxs00 | (idxs10 << 2)
    meta_41 = idxs01 | (idxs11 << 2)
    meta_42 = idxs02 | (idxs12 << 2)
    meta_43 = idxs03 | (idxs13 << 2)
    meta = (
            meta_40
            | (meta_41 << 4)
            | (meta_42 << 8)
            | (meta_43 << 12)
    )

    group, interweave = 32, 4

    dest_row = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8
    if dest_row % 2 == 0:
        dest_row_ = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8 + tl.arange(0, BLOCK_SIZE // 16) % 2
        dest_col_ = tl.arange(0, BLOCK_SIZE // 16) // 2 * 2
        index = (dest_col_ // 2) * m * 2 + dest_row_ * 2 + dest_col_ % 2
        tl.store(meta_reordered_ptr + index, meta, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    else:
        dest_row_ = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8 - (
                tl.arange(0, BLOCK_SIZE // 16) + 1) % 2
        dest_col_ = tl.arange(0, BLOCK_SIZE // 16) // 2 * 2 + 1
        index = (dest_col_ // 2) * m * 2 + dest_row_ * 2 + dest_col_ % 2
        tl.store(meta_reordered_ptr + index, meta, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)


def _sparse_semi_structured_from_dense_triton(dense, sparse, meta, mask: Optional[Tensor] = None,
                                              dtype: Optional[torch.dtype] = None):
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device
    BLOCK_SIZE = triton.next_power_of_2(k)

    if dense.dtype != torch.half and dtype != torch.half:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    if m % 32 != 0:
        raise RuntimeError(
            f"Number rows columns of dense matrix {m} must be divisible by 32"
        )
    if k % 16 != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {16}"
        )
    # num_warps = 4
    # if BLOCK_SIZE >= 2048:
    #     num_warps = 8
    # if BLOCK_SIZE >= 4096:
    #     num_warps = 16

    num_warps = 2
    if BLOCK_SIZE >= 2048:
        num_warps = 4
    if BLOCK_SIZE >= 4096:
        num_warps = 8
    if BLOCK_SIZE >= 8192:
        num_warps = 16

    # sparse, meta_reordered = torch.empty(m, k // 2, device=device, dtype=torch.float16), \
    #     torch.empty(m, k // 16, device=device, dtype=torch.int16)

    _sparse_semi_structured_from_dense_kernel[(m,)](
        dense,
        sparse,
        meta,
        dense.stride(0),
        sparse.stride(0),
        dense.stride(1),
        m, k,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    ) if mask is None else _sparse_semi_structured_from_dense_with_mask_kernel[(m,)](
        dense,
        sparse,
        meta,
        mask,
        dense.stride(0),
        sparse.stride(0),
        mask.stride(0),
        dense.stride(1),
        mask.stride(1),
        m, k,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (sparse, meta)


@triton.jit
def _sparse_semi_structured_from_dense_MVUE12_kernel(
        dense_ptr,
        sparse_ptr,
        meta_reordered_ptr,
        dense_row_stride,
        sparse_row_stride,
        dense_col_stride,
        m, k,  # dense.shape
        seeds,
        BLOCK_SIZE: tl.constexpr,

):
    row_idx = tl.program_id(0)

    col_idx = 16 * tl.arange(0, BLOCK_SIZE // 16)
    m00 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))  # A0
    col_idx += 1
    m10 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))  # B0
    col_idx += 1
    m20 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m30 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))

    col_idx += 1
    m01 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))  # A1
    col_idx += 1
    m11 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m21 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m31 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))

    col_idx += 1
    m02 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m12 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m22 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m32 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))

    col_idx += 1
    m03 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m13 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m23 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))
    col_idx += 1
    m33 = tl.load(dense_ptr + row_idx * dense_row_stride + col_idx * dense_col_stride,
                  mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                  other=-float('inf'))

    left0 = tl.abs(m00) + 6e-8
    left1 = tl.abs(m20) + 6e-8
    left2 = tl.abs(m01) + 6e-8
    left3 = tl.abs(m21) + 6e-8
    left4 = tl.abs(m02) + 6e-8
    left5 = tl.abs(m22) + 6e-8
    left6 = tl.abs(m03) + 6e-8
    left7 = tl.abs(m23) + 6e-8
    right0 = tl.abs(m10) + 6e-8
    right1 = tl.abs(m30) + 6e-8
    right2 = tl.abs(m11) + 6e-8
    right3 = tl.abs(m31) + 6e-8
    right4 = tl.abs(m12) + 6e-8
    right5 = tl.abs(m32) + 6e-8
    right6 = tl.abs(m13) + 6e-8
    right7 = tl.abs(m33) + 6e-8

    m0 = left0 + right0  # row_
    m1 = left1 + right1
    m2 = left2 + right2
    m3 = left3 + right3
    m4 = left4 + right4
    m5 = left5 + right5
    m6 = left6 + right6
    m7 = left7 + right7

    p0 = left0 / m0
    p1 = left1 / m1
    p2 = left2 / m2
    p3 = left3 / m3
    p4 = left4 / m4
    p5 = left5 / m5
    p6 = left6 / m6
    p7 = left7 / m7

    seed0 = tl.load(seeds + row_idx * 8)
    seed1 = tl.load(seeds + row_idx * 8 + 1)
    seed2 = tl.load(seeds + row_idx * 8 + 2)
    seed3 = tl.load(seeds + row_idx * 8 + 3)
    seed4 = tl.load(seeds + row_idx * 8 + 4)
    seed5 = tl.load(seeds + row_idx * 8 + 5)
    seed6 = tl.load(seeds + row_idx * 8 + 6)
    seed7 = tl.load(seeds + row_idx * 8 + 7)

    random0 = tl.rand(seed0, tl.arange(0, BLOCK_SIZE // 16))
    random1 = tl.rand(seed1, tl.arange(0, BLOCK_SIZE // 16))
    random2 = tl.rand(seed2, tl.arange(0, BLOCK_SIZE // 16))
    random3 = tl.rand(seed3, tl.arange(0, BLOCK_SIZE // 16))
    random4 = tl.rand(seed4, tl.arange(0, BLOCK_SIZE // 16))
    random5 = tl.rand(seed5, tl.arange(0, BLOCK_SIZE // 16))
    random6 = tl.rand(seed6, tl.arange(0, BLOCK_SIZE // 16))
    random7 = tl.rand(seed7, tl.arange(0, BLOCK_SIZE // 16))

    m00_ = (random0 <= p0)
    m20_ = (random1 <= p1)
    m01_ = (random2 <= p2)
    m21_ = (random3 <= p3)
    m02_ = (random4 <= p4)
    m22_ = (random5 <= p5)
    m03_ = (random6 <= p6)
    m23_ = (random7 <= p7)
    m10_ = (random0 > p0)
    m30_ = (random1 > p1)
    m11_ = (random2 > p2)
    m31_ = (random3 > p3)
    m12_ = (random4 > p4)
    m32_ = (random5 > p5)
    m13_ = (random6 > p6)
    m33_ = (random7 > p7)

    left0_ = tl.where(m00_, m00, 0.0)
    right0_ = tl.where(m10_, m10, 0.0)
    left1_ = tl.where(m20_, m20, 0.0)
    right1_ = tl.where(m30_, m30, 0.0)
    left2_ = tl.where(m01_, m01, 0.0)
    right2_ = tl.where(m11_, m11, 0.0)
    left3_ = tl.where(m21_, m21, 0.0)
    right3_ = tl.where(m31_, m31, 0.0)
    left4_ = tl.where(m02_, m02, 0.0)
    right4_ = tl.where(m12_, m12, 0.0)
    left5_ = tl.where(m22_, m22, 0.0)
    right5_ = tl.where(m32_, m32, 0.0)
    left6_ = tl.where(m03_, m03, 0.0)
    right6_ = tl.where(m13_, m13, 0.0)
    left7_ = tl.where(m23_, m23, 0.0)
    right7_ = tl.where(m33_, m33, 0.0)

    left_output0 = left0_ / left0 * m0
    left_output1 = left1_ / left1 * m1
    left_output2 = left2_ / left2 * m2
    left_output3 = left3_ / left3 * m3
    left_output4 = left4_ / left4 * m4
    left_output5 = left5_ / left5 * m5
    left_output6 = left6_ / left6 * m6
    left_output7 = left7_ / left7 * m7

    right_output0 = right0_ / right0 * m0
    right_output1 = right1_ / right1 * m1
    right_output2 = right2_ / right2 * m2
    right_output3 = right3_ / right3 * m3
    right_output4 = right4_ / right4 * m4
    right_output5 = right5_ / right5 * m5
    right_output6 = right6_ / right6 * m6
    right_output7 = right7_ / right7 * m7

    m00 = left_output0
    m20 = left_output1
    m01 = left_output2
    m21 = left_output3
    m02 = left_output4
    m22 = left_output5
    m03 = left_output6
    m23 = left_output7
    m10 = right_output0
    m30 = right_output1
    m11 = right_output2
    m31 = right_output3
    m12 = right_output4
    m32 = right_output5
    m13 = right_output6
    m33 = right_output7

    # x10, x20, x30, x40, x50, x60 = tl.abs(m00) > tl.abs(m10), tl.abs(m00) > tl.abs(m20), tl.abs(m00) > tl.abs(
    #     m30), tl.abs(m10) > tl.abs(m20), tl.abs(m10) > tl.abs(m30), tl.abs(m20) > tl.abs(m30)
    # m00_, m10_, m20_, m30_ = x20 & x30 | x10 & x20 | x10 & x30, ~x10 & x50 | x40 & x50 | ~x10 & x40, ~x20 & ~x40 | ~x20 & x60 | ~x40 & x60, ~x30 & ~x50 | ~x30 & ~x60 | ~x50 & ~x60
    #
    # x11, x21, x31, x41, x51, x61 = tl.abs(m01) > tl.abs(m11), tl.abs(m01) > tl.abs(m21), tl.abs(m01) > tl.abs(
    #     m31), tl.abs(m11) > tl.abs(m21), tl.abs(m11) > tl.abs(m31), tl.abs(m21) > tl.abs(m31)
    # m01_, m11_, m21_, m31_ = x21 & x31 | x11 & x21 | x11 & x31, ~x11 & x51 | x41 & x51 | ~x11 & x41, ~x21 & ~x41 | ~x21 & \
    #                          x61 | ~x41 & x61, ~x31 & ~x51 | ~x31 & ~x61 | ~x51 & ~x61
    #
    # x12, x22, x32, x42, x52, x62 = tl.abs(m02) > tl.abs(m12), tl.abs(m02) > tl.abs(m22), tl.abs(m02) > tl.abs(
    #     m32), tl.abs(m12) > tl.abs(m22), tl.abs(m12) > tl.abs(m32), tl.abs(m22) > tl.abs(m32)
    # m02_, m12_, m22_, m32_ = x22 & x32 | x12 & x22 | x12 & x32, ~x12 & x52 | x42 & x52 | ~x12 & x42, ~x22 & ~x42 | ~x22 & \
    #                          x62 | ~x42 & x62, ~x32 & ~x52 | ~x32 & ~x62 | ~x52 & ~x62
    #
    # x13, x23, x33, x43, x53, x63 = tl.abs(m03) > tl.abs(m13), tl.abs(m03) > tl.abs(m23), tl.abs(m03) > tl.abs(
    #     m33), tl.abs(m13) > tl.abs(m23), tl.abs(m13) > tl.abs(m33), tl.abs(m23) > tl.abs(m33)
    # m03_, m13_, m23_, m33_ = x23 & x33 | x13 & x23 | x13 & x33, ~x13 & x53 | x43 & x53 | ~x13 & x43, ~x23 & ~x43 | ~x23 & \
    #                          x63 | ~x43 & x63, ~x33 & ~x53 | ~x33 & ~x63 | ~x53 & ~x63

    # initial codes are as bellow.
    # m00_, m10_, m20_, m30_ = m00.to(tl.int1), m10.to(tl.int1), m20.to(tl.int1), m30.to(tl.int1)
    # m01_, m11_, m21_, m31_ = m01.to(tl.int1), m11.to(tl.int1), m21.to(tl.int1), m31.to(tl.int1)
    # m02_, m12_, m22_, m32_ = m02.to(tl.int1), m12.to(tl.int1), m22.to(tl.int1), m32.to(tl.int1)
    # m03_, m13_, m23_, m33_ = m03.to(tl.int1), m13.to(tl.int1), m23.to(tl.int1), m33.to(tl.int1)

    bit00 = ~m00_ & m10_
    bit10 = ~m00_ & ~m10_
    bit20 = bit10 | ~m20_
    bit30 = bit00 | ~m10_ | m20_
    idxs00 = bit00 | (bit10.to(tl.int64) << 1)
    idxs10 = bit20 | (bit30.to(tl.int64) << 1)
    sparse00 = tl.where(bit10, tl.where(bit00, m30, m20), tl.where(bit00, m10, m00))
    sparse10 = tl.where(bit30, tl.where(bit20, m30, m20), tl.where(bit20, m10, m00))

    bit01 = ~m01_ & m11_
    bit11 = ~m01_ & ~m11_
    bit21 = bit11 | ~m21_
    bit31 = bit01 | ~m11_ | m21_
    idxs01 = bit01 | (bit11.to(tl.int64) << 1)
    idxs11 = bit21 | (bit31.to(tl.int64) << 1)
    sparse01 = tl.where(bit11, tl.where(bit01, m31, m21), tl.where(bit01, m11, m01))
    sparse11 = tl.where(bit31, tl.where(bit21, m31, m21), tl.where(bit21, m11, m01))

    bit02 = ~m02_ & m12_
    bit12 = ~m02_ & ~m12_
    bit22 = bit12 | ~m22_
    bit32 = bit02 | ~m12_ | m22_
    idxs02 = bit02 | (bit12.to(tl.int64) << 1)
    idxs12 = bit22 | (bit32.to(tl.int64) << 1)
    sparse02 = tl.where(bit12, tl.where(bit02, m32, m22), tl.where(bit02, m12, m02))
    sparse12 = tl.where(bit32, tl.where(bit22, m32, m22), tl.where(bit22, m12, m02))

    bit03 = ~m03_ & m13_
    bit13 = ~m03_ & ~m13_
    bit23 = bit13 | ~m23_
    bit33 = bit03 | ~m13_ | m23_
    idxs03 = bit03 | (bit13.to(tl.int64) << 1)
    idxs13 = bit23 | (bit33.to(tl.int64) << 1)
    sparse03 = tl.where(bit13, tl.where(bit03, m33, m23), tl.where(bit03, m13, m03))
    sparse13 = tl.where(bit33, tl.where(bit23, m33, m23), tl.where(bit23, m13, m03))

    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16), sparse00,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 1, sparse10,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 1 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 2, sparse01,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 2 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 3, sparse11,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 3 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 4, sparse02,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 4 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 5, sparse12,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 5 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 6, sparse03,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 6 < k / 2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + 8 * tl.arange(0, BLOCK_SIZE // 16) + 7, sparse13,
             mask=8 * tl.arange(0, BLOCK_SIZE // 16) + 7 < k / 2)

    meta_40 = idxs00 | (idxs10 << 2)
    meta_41 = idxs01 | (idxs11 << 2)
    meta_42 = idxs02 | (idxs12 << 2)
    meta_43 = idxs03 | (idxs13 << 2)
    meta = (
            meta_40
            | (meta_41 << 4)
            | (meta_42 << 8)
            | (meta_43 << 12)
    )

    group, interweave = 32, 4

    dest_row = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8
    if dest_row % 2 == 0:
        dest_row_ = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8 + tl.arange(0, BLOCK_SIZE // 16) % 2
        dest_col_ = tl.arange(0, BLOCK_SIZE // 16) // 2 * 2
        index = (dest_col_ // 2) * m * 2 + dest_row_ * 2 + dest_col_ % 2
        tl.store(meta_reordered_ptr + index, meta, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    else:
        dest_row_ = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8 - (
                tl.arange(0, BLOCK_SIZE // 16) + 1) % 2
        dest_col_ = tl.arange(0, BLOCK_SIZE // 16) // 2 * 2 + 1
        index = (dest_col_ // 2) * m * 2 + dest_row_ * 2 + dest_col_ % 2
        tl.store(meta_reordered_ptr + index, meta, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)


def _sparse_semi_structured_from_dense_triton_MVUE12(dense, sparse, meta):
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device
    BLOCK_SIZE = triton.next_power_of_2(k)

    if dense.dtype != torch.half:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    if m % 32 != 0:
        raise RuntimeError(
            f"Number rows columns of dense matrix {m} must be divisible by 32"
        )
    if k % 16 != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {16}"
        )

    # num_warps = 4
    # if BLOCK_SIZE >= 2048:
    #     num_warps = 8
    # if BLOCK_SIZE >= 4096:
    #     num_warps = 16

    num_warps = 2
    if BLOCK_SIZE >= 2048:
        num_warps = 4
    if BLOCK_SIZE >= 4096:
        num_warps = 8
    if BLOCK_SIZE >= 8192:
        num_warps = 16

    # sparse, meta_reordered = torch.empty(m, k // 2, device=device, dtype=torch.float16), \
    #     torch.empty(m, k // 16, device=device, dtype=torch.int16)
    seeds = torch.randint(0, 2 ** 31 - 1, (m, 8), device='cuda')
    _sparse_semi_structured_from_dense_MVUE12_kernel[(m,)](
        dense,
        sparse,
        meta,
        dense.stride(0),
        sparse.stride(0),
        dense.stride(1),
        m, k,
        seeds,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return (sparse, meta)


@triton.jit
def _sparse_semi_structured_to_dense_kernel(
        sparse_ptr,
        meta_reordered_ptr,
        dense_ptr,
        m, k,  # dense.shape
        BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)

    group, interweave = 32, 4
    dest_row = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8
    if dest_row % 2 == 0:
        dest_row_ = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8 + tl.arange(0, BLOCK_SIZE // 16) % 2
        dest_col_ = tl.arange(0, BLOCK_SIZE // 16) // 2 * 2
        index = (dest_col_ // 2) * m * 2 + dest_row_ * 2 + dest_col_ % 2
        meta = tl.load(meta_reordered_ptr + index, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                       other=-float('inf'))  # shape=k//16
    else:
        dest_row_ = row_idx // 32 * 32 + (row_idx % 8) * 4 + (row_idx % group) // 8 - (
                tl.arange(0, BLOCK_SIZE // 16) + 1) % 2
        dest_col_ = tl.arange(0, BLOCK_SIZE // 16) // 2 * 2 + 1
        index = (dest_col_ // 2) * m * 2 + dest_row_ * 2 + dest_col_ % 2
        meta = tl.load(meta_reordered_ptr + index, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16,
                       other=-float('inf'))  # shape=k//16

    meta_20 = (meta & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16))
    meta_21 = ((meta >> 2) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16))
    meta_22 = ((meta >> 4) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 4)
    meta_23 = ((meta >> 6) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 4)
    meta_24 = ((meta >> 8) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 8)
    meta_25 = ((meta >> 10) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 8)
    meta_26 = ((meta >> 12) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 12)
    meta_27 = ((meta >> 14) & 0b11) + (row_idx * k + 16 * tl.arange(0, BLOCK_SIZE // 16) + 12)

    row0 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16),
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row1 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 1,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row2 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 2,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row3 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 3,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row4 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 4,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row5 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 5,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row6 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 6,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))
    row7 = tl.load(sparse_ptr + row_idx * k // 2 + 8 * tl.arange(0, BLOCK_SIZE // 16) + 7,
                   mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16, other=-float('inf'))

    tl.store(dense_ptr + meta_20, row0, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_21, row1, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_22, row2, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_23, row3, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_24, row4, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_25, row5, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_26, row6, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)
    tl.store(dense_ptr + meta_27, row7, mask=tl.arange(0, BLOCK_SIZE // 16) < k // 16)


def _sparse_semi_structured_to_dense_triton(sparse, meta_reordered):
    assert sparse.is_contiguous()
    assert meta_reordered.is_contiguous()
    if sparse.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor"
        )

    m, k = sparse.shape[0], sparse.shape[1] * 2
    device = sparse.device

    if meta_reordered.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor"
        )
    if meta_reordered.device != device:
        raise RuntimeError(
            f"Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device"
        )

    meta_dtype = meta_reordered.dtype
    if meta_dtype is not torch.int16:
        raise RuntimeError(f"Invalid datatype {meta_dtype} of meta matrix")

    BLOCK_SIZE = triton.next_power_of_2(k)

    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16

    # num_warps = 2
    # if BLOCK_SIZE >= 2048:
    #     num_warps = 4
    # if BLOCK_SIZE >= 4096:
    #     num_warps = 8
    # if BLOCK_SIZE >= 8192:
    #     num_warps = 16

    dense = torch.zeros((m, k), dtype=torch.half, device=device)
    _sparse_semi_structured_to_dense_kernel[(m,)](
        sparse,
        meta_reordered,
        dense,
        m, k,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dense
