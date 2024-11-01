import torch


def get_device():
    # if torch.backends.mps.is_available():
    #     return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'
