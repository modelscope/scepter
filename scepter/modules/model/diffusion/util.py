import torch

def _i(tensor, t, x):
    """
    Index tensor using t and format the output according to x.
    """
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    if isinstance(t, torch.Tensor):
        t = t.to(tensor.device)
    return tensor[t].view(shape).to(x.device)