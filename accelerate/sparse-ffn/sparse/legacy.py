import torch


def sparse12(weight):
    N, M = 1, 2

    output = weight.clone()
    length = weight.numel()
    group = int(length / M)

    weight_temp = weight.detach().abs().reshape(group, M)
    index = torch.argsort(weight_temp, dim=1)[:, :int(M - N)]

    w_b = torch.ones(weight_temp.shape, device=weight_temp.device, dtype=torch.int)
    w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)

    return output * w_b, w_b


def sparse24(weight):
    N, M = 2, 4

    output = weight.clone()
    length = weight.numel()
    group = int(length / M)

    weight_temp = weight.detach().abs().reshape(group, M)
    index = torch.argsort(weight_temp, dim=1)[:, :int(M - N)]

    w_b = torch.ones(weight_temp.shape, device=weight_temp.device, dtype=torch.int)
    w_b = w_b.scatter_(dim=1, index=index, value=0).reshape(weight.shape)

    return output * w_b, w_b


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


def MVUE24_approx(weight):
    eps = 1.19209e-07
    device = weight.device
    shape = weight.shape
    weight = weight.reshape(-1, 4)
    sign = torch.sign(weight)
    weight = weight.abs() + eps
    sum = torch.sum(weight, dim=1, keepdim=True)

    p = weight / sum

    weight0 = weight.clone()
    weight0[:, 0:1] = 0
    sum0 = torch.sum(weight0, dim=1, keepdim=True)
    p0 = p[:, 0:1] * weight0 / sum0

    weight1 = weight.clone()
    weight1[:, 1:2] = 0
    sum1 = torch.sum(weight1, dim=1, keepdim=True)
    p1 = p[:, 1:2] * weight1 / sum1

    weight2 = weight.clone()
    weight2[:, 2:3] = 0
    sum2 = torch.sum(weight2, dim=1, keepdim=True)
    p2 = p[:, 2:3] * weight2 / sum2

    weight3 = weight.clone()
    weight3[:, 3:4] = 0
    sum3 = torch.sum(weight3, dim=1, keepdim=True)
    p3 = p[:, 3:4] * weight3 / sum3

    p = p + p0 + p1 + p2 + p3
    weight_ = weight.clone()
    index0 = torch.multinomial(weight, 1)
    weight.scatter_(1, index0, 0)
    index1 = torch.multinomial(weight, 1)
    weight.scatter_(1, index1, 0)
    weight = weight_ - weight
    weight = weight / p
    weight = weight * sign
    weight = weight.view(shape)
    return weight


def transposable_sparse(weight, abs=True):
    output = weight.clone()
    weight_temp = weight.detach().abs() if abs is True else weight.detach()
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

    return output * w_b, w_b
