import random

import torch
import triton
import triton.language as tl


def sparse12(weight):
    N, M = 1, 2

    output = weight.clone()
    length = weight.numel()
    group = int(length / M)

    weight_temp = weight.detach().abs().reshape(group, M)
    index = torch.argsort(weight_temp, dim=1)[:, :int(M - N)]

    w_b = torch.ones(weight_temp.shape, device=weight_temp.device, dtype=torch.int)
    w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)

    return output * w_b


def sparse24(weight):
    N, M = 2, 4

    output = weight.clone()
    length = weight.numel()
    group = int(length / M)

    weight_temp = weight.detach().abs().reshape(group, M)
    index = torch.argsort(weight_temp, dim=1)[:, :int(M - N)]

    w_b = torch.ones(weight_temp.shape, device=weight_temp.device, dtype=torch.int)
    w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)

    return output * w_b


def MVUE12(weight):
    device = weight.device
    shape = weight.shape
    weight = weight.reshape(-1, 2)
    sign = torch.sign(weight)
    weight = weight.abs()

    sum = torch.sum(weight, dim=1).view(-1, 1)
    p = weight + 6e-8
    index = torch.multinomial(p, 1).view(-1)

    L12 = torch.tensor([[1, 0], [0, 1]], device=weight.device)

    mask = torch.index_select(L12, 0, index)
    weight = mask * sum
    weight = weight * sign
    weight = weight.view(shape)
    return weight


def MVUE24(weight):
    device = weight.device
    shape = weight.shape
    weight = weight.reshape(-1, 4)
    sign = torch.sign(weight)
    weight = weight.abs()

    a = torch.argsort(weight)
    c = torch.argsort(a)
    sum = torch.sum(weight, dim=1).view(-1, 1)
    b = torch.gather(weight, 1, a)

    a1 = b[:, 0]
    a2 = b[:, 1]
    a3 = b[:, 2]
    a4 = b[:, 3]
    flag1 = 2 * a1 + a3 - a4
    flag2 = a1 + a2 + a3 - a4

    p12 = torch.zeros_like(a1)
    p13 = (2 * a1 + a3 - a4) / (2 * (a1 + a2 + a3 + a4))
    p14 = (2 * a1 - a3 + a4) / (2 * (a1 + a2 + a3 + a4))
    p23 = (2 * a2 + a3 - a4) / (2 * (a1 + a2 + a3 + a4))
    p24 = (2 * a2 - a3 + a4) / (2 * (a1 + a2 + a3 + a4))
    p34 = (-a1 - a2 + a3 + a4) / (a1 + a2 + a3 + a4)
    p1 = torch.stack((p12, p13, p14, p23, p24, p34), dim=1)

    p12 = torch.zeros_like(a1)
    p13 = torch.zeros_like(a1)
    p14 = (2 * a1) / (a1 + a2 + a3 + a4)
    p23 = (a1 + a2 + a3 - a4) / (a1 + a2 + a3 + a4)
    p24 = (-a1 + a2 - a3 + a4) / (2 * (a1 + a2 + a3 + a4))
    p34 = (-a1 - a2 + a3 + a4) / (a1 + a2 + a3 + a4)
    p2 = torch.stack((p12, p13, p14, p23, p24, p34), dim=1)

    p12 = torch.zeros_like(a1)
    p13 = torch.zeros_like(a1)
    p14 = a1 / (a1 + a2 + a3)
    p23 = torch.zeros_like(a1)
    p24 = a2 / (a1 + a2 + a3)
    p34 = a3 / (a1 + a2 + a3)
    p3 = torch.stack((p12, p13, p14, p23, p24, p34), dim=1)

    bool1 = (flag1 > 0)
    bool2 = (flag2 > 0)
    hi = ~(bool1 | ((~bool1) & bool2))
    lo = (~bool1) & bool2
    index = 2 * hi + lo
    p_ = torch.stack((p1, p2, p3), dim=1)
    index = torch.stack([index] * 6, dim=1).view(-1, 1, 6)
    p = torch.gather(p_, 1, index).view(-1, 6)

    p = torch.where(p < 0, torch.full_like(p, 0), p)
    p = torch.where(torch.isnan(p), torch.full_like(p, 0), p)
    p = torch.where(torch.isinf(p), torch.full_like(p, 1), p)
    p += 6e-8
    index = torch.multinomial(p, 1).view(-1)

    L24 = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]],
                       device=weight.device)

    mask = torch.index_select(L24, 0, index)
    weight = mask * sum / 2
    weight = torch.gather(weight, 1, c)
    weight = weight * sign
    weight = weight.view(shape)
    return weight


def transposable_sparse(weight):
    output = weight.clone()
    weight_temp = weight.detach().abs()
    M = 4
    a = weight_temp
    shape = a.shape
    b = torch.chunk(a, a.shape[1] // M, dim=1)
    c = torch.concat(b, dim=0)
    f = c.view(-1, 16)

    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                          0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                          1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0,
                          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                          1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                          0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                          1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                          0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                         [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1,
                          1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
                          0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                          1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                          1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
                          0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1,
                          1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
                         [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
                          0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
                          1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0,
                          0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                         [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
                          0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0,
                          0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
                          1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                         [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
                          1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1,
                          1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
                          0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                          0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
                          1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1,
                          1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                         [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                          1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1,
                          0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
                          0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                          1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0,
                          0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1,
                          1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                         [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                          0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                          1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0,
                          0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]], device=weight.device,
                        dtype=weight.dtype)
    o = f @ mask
    p = torch.argmax(o, dim=1)
    q = torch.index_select(mask, dim=1, index=p).T
    g = q.reshape(-1, M)
    h = torch.chunk(g, shape[1] // M, dim=0)
    i = torch.concat(h, dim=1)
    w_b = i

    return output * w_b


def transposable_sparse_mask(weight, abs=False):
    weight_temp = weight.detach() if abs == False else weight.detach().abs()
    M = 4
    a = weight_temp
    shape = a.shape
    b = torch.chunk(a, a.shape[1] // M, dim=1)
    c = torch.concat(b, dim=0)
    f = c.view(-1, 16)

    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                          0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                          1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0,
                          0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                          1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                         [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
                          0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
                          1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
                          0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
                         [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1,
                          1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
                          0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
                          1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
                          1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,
                          0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1,
                          1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1],
                         [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
                          0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
                          1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0,
                          0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
                         [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1,
                          0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0,
                          0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
                          1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                         [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0,
                          1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1,
                          1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
                          0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                          0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
                          1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1,
                          1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1],
                         [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
                          1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1,
                          0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
                          0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
                         [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                          1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0,
                          0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1,
                          1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                         [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1,
                          0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1,
                          1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0,
                          0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]], device=weight.device,
                        dtype=weight.dtype)
    o = f @ mask
    p = torch.argmax(o, dim=1)
    q = torch.index_select(mask, dim=1, index=p).T
    g = q.reshape(-1, M)
    h = torch.chunk(g, shape[1] // M, dim=0)
    i = torch.concat(h, dim=1)
    w_b = i

    return w_b


@triton.jit
def _MVUE24_approx(x0, x1, x2, x3,
                   random0, random1):
    eps = 1.19209e-07
    a0 = tl.abs(x0) + eps
    a1 = tl.abs(x1) + eps
    a2 = tl.abs(x2) + eps
    a3 = tl.abs(x3) + eps
    sum = a0 + a1 + a2 + a3

    t0 = a0 / sum
    t1 = a1 / sum
    t2 = a2 / sum
    t3 = a3 / sum

    s0 = sum - a0
    s1 = sum - a1
    s2 = sum - a2
    s3 = sum - a3

    k0 = t0 / s0
    k1 = t1 / s1
    k2 = t2 / s2
    k3 = t3 / s3
    k = k0 + k1 + k2 + k3

    p0 = (t0 + a0 * (k - k0))
    p1 = (t1 + a1 * (k - k1))
    p2 = (t2 + a2 * (k - k2))
    p3 = (t3 + a3 * (k - k3))

    m0 = (random0 <= t0)
    m1 = ((random0 <= (t0 + t1)) & ~m0)
    m2 = ((random0 <= (t0 + t1 + t2)) & ~m1 & ~m0)
    m3 = ~m2 & ~m1 & ~m0

    d_a0 = ~m0 * a0
    d_a1 = ~m1 * a1
    d_a2 = ~m2 * a2
    d_a3 = ~m3 * a3
    d_sum = d_a0 + d_a1 + d_a2 + d_a3

    t = random1 * d_sum
    d_m0 = (t <= d_a0)
    d_m1 = ((t <= (d_a0 + d_a1)) & ~d_m0)
    d_m2 = ((t <= (d_a0 + d_a1 + d_a2)) & ~d_m1 & ~d_m0)
    d_m3 = ~d_m2 & ~d_m1 & ~d_m0

    m0, m1, m2, m3 = m0 | d_m0, m1 | d_m1, m2 | d_m2, m3 | d_m3
    a0 = x0 / p0
    a1 = x1 / p1
    a2 = x2 / p2
    a3 = x3 / p3

    return a0, a1, a2, a3, m0, m1, m2, m3


def get_configs():
    configs = []
    for block in [32, 64, 128]:
        for num_stages in [2, 3, 4, 5]:
            for num_warps in [2, 4, 8]:
                configs.append(triton.Config({'BLOCK_SIZE': block}, num_stages=num_stages, num_warps=num_warps))

    return configs


@triton.autotune(
    configs=get_configs(),
    key=['m', 'k'],
)
@triton.jit
def _MVUE24_approx_triton(
        dense_ptr,
        sparse_ptr,
        dense_row_stride,
        sparse_row_stride,
        dense_col_stride,
        sparse_col_stride,
        m, k,
        seed,
        BLOCK_SIZE: tl.constexpr,
        ARRAY_LAYOUT: tl.constexpr
):
    if ARRAY_LAYOUT == 'row':
        row_idx = tl.program_id(0)
        col_idx = tl.program_id(1) * 16 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) * 16
        mask = col_idx < k
    elif ARRAY_LAYOUT == 'col':
        row_idx = tl.arange(0, BLOCK_SIZE) + tl.program_id(0) * BLOCK_SIZE
        col_idx = tl.program_id(1) * 16
        mask = row_idx < m
    dense_40 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 0) * dense_col_stride, mask=mask)
    dense_41 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 1) * dense_col_stride, mask=mask)
    dense_42 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 2) * dense_col_stride, mask=mask)
    dense_43 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 3) * dense_col_stride, mask=mask)
    dense_44 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 4) * dense_col_stride, mask=mask)
    dense_45 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 5) * dense_col_stride, mask=mask)
    dense_46 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 6) * dense_col_stride, mask=mask)
    dense_47 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 7) * dense_col_stride, mask=mask)
    dense_48 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 8) * dense_col_stride, mask=mask)
    dense_49 = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 9) * dense_col_stride, mask=mask)
    dense_4A = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 10) * dense_col_stride, mask=mask)
    dense_4B = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 11) * dense_col_stride, mask=mask)
    dense_4C = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 12) * dense_col_stride, mask=mask)
    dense_4D = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 13) * dense_col_stride, mask=mask)
    dense_4E = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 14) * dense_col_stride, mask=mask)
    dense_4F = tl.load(dense_ptr + row_idx * dense_row_stride + (col_idx + 15) * dense_col_stride, mask=mask)

    if ARRAY_LAYOUT == 'row':
        seed0 = seed + (tl.program_id(0) + tl.program_id(1) * m) * 2
        seed1 = seed + (tl.program_id(0) + tl.program_id(1) * m) * 2 + 1
    else:
        seed0 = seed + (tl.program_id(0) * k // 16 + tl.program_id(1)) * 2
        seed1 = seed + (tl.program_id(0) * k // 16 + tl.program_id(1)) * 2 + 1
    random0, random1, random2, random3 = tl.rand4x(seed0, tl.arange(0, BLOCK_SIZE), n_rounds=5)
    random4, random5, random6, random7 = tl.rand4x(seed1, tl.arange(0, BLOCK_SIZE), n_rounds=5)

    dense_40, dense_41, dense_42, dense_43, m0, m1, m2, m3 = _MVUE24_approx(dense_40, dense_41, dense_42, dense_43,
                                                                            random0, random1)
    dense_44, dense_45, dense_46, dense_47, m4, m5, m6, m7 = _MVUE24_approx(dense_44, dense_45, dense_46, dense_47,
                                                                            random2, random3)
    dense_48, dense_49, dense_4A, dense_4B, m8, m9, mA, mB = _MVUE24_approx(dense_48, dense_49, dense_4A, dense_4B,
                                                                            random4, random5)
    dense_4C, dense_4D, dense_4E, dense_4F, mC, mD, mE, mF = _MVUE24_approx(dense_4C, dense_4D, dense_4E, dense_4F,
                                                                            random6, random7)

    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 0) * sparse_col_stride, dense_40, mask=mask & m0)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 1) * sparse_col_stride, dense_41, mask=mask & m1)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 2) * sparse_col_stride, dense_42, mask=mask & m2)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 3) * sparse_col_stride, dense_43, mask=mask & m3)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 4) * sparse_col_stride, dense_44, mask=mask & m4)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 5) * sparse_col_stride, dense_45, mask=mask & m5)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 6) * sparse_col_stride, dense_46, mask=mask & m6)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 7) * sparse_col_stride, dense_47, mask=mask & m7)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 8) * sparse_col_stride, dense_48, mask=mask & m8)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 9) * sparse_col_stride, dense_49, mask=mask & m9)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 10) * sparse_col_stride, dense_4A, mask=mask & mA)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 11) * sparse_col_stride, dense_4B, mask=mask & mB)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 12) * sparse_col_stride, dense_4C, mask=mask & mC)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 13) * sparse_col_stride, dense_4D, mask=mask & mD)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 14) * sparse_col_stride, dense_4E, mask=mask & mE)
    tl.store(sparse_ptr + row_idx * sparse_row_stride + (col_idx + 15) * sparse_col_stride, dense_4F, mask=mask & mF)


def MVUE24_approx_triton(dense):
    m, k = dense.shape
    device = dense.device
    seed = random.randint(0, 2 ** 31 - 1)
    sparse = torch.zeros_like(dense)

    row_stride, col_stride = dense.stride()
    if row_stride > col_stride:
        array_layout = 'row'
        grid = lambda META: (m, triton.cdiv(k, 16 * META['BLOCK_SIZE']))
    else:
        array_layout = 'col'
        grid = lambda META: (triton.cdiv(m, META['BLOCK_SIZE']), k // 16,)
    func = _MVUE24_approx_triton
    func[grid](
        dense,
        sparse,
        dense.stride(0),
        sparse.stride(0),
        dense.stride(1),
        sparse.stride(1),
        m, k,
        seed,
        ARRAY_LAYOUT=array_layout
    )
    return sparse
