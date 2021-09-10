import torch


def joint_distance(pred, truth, reduction='mean'):
    batch_size = pred.size(0)

    # Reshape to pose format
    pred = pred.view((batch_size, -1, 3))
    truth = truth.view((batch_size, -1, 3))

    diff = (pred - truth).norm(dim=-1)

    if reduction == 'none':
        return diff
    elif reduction == 'mean':
        return torch.mean(diff)
    else:
        raise "Reduction mode " + reduction + " not found"
