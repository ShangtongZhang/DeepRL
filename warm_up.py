import torch

def hook(gpus):
    ns = [torch.rand((1000, 1000)).cuda(gpu) for gpu in gpus]
    while True:
        for n in ns:
            n.add_(0.01)

if __name__ == '__main__':
    gpus = [0, 1, 2, 3]
    hook(gpus)