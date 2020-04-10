import torch


def mellow_max(x, r, b=None):
    n = len(x)
    if b is None:
        shift = torch.stack(x).max(0)[0]
    else:
        shift = b
    mellow_max_loss = 0
    for i in range(n):
        mellow_max_loss += torch.exp(x[i] - shift)
    if b is None:
        mellow_max_loss = torch.log(mellow_max_loss) / r
    else:
        mellow_max_loss = (torch.log(mellow_max_loss) + shift) / r
    return mellow_max_loss


def reorder_and_match(source, target_order):
    new_target = torch.ones_like(source)
    for i in range(target_order.shape[0]):
        for j in range(target_order.shape[1]):
            new_target[target_order[i, j][0], j] = source[i, j]
    return new_target
