import torch
import numpy as np
from torch import nn

def create_logloss_label(label_size, rPos):
    label = torch.ones(1, label_size, label_size)
    center = np.array([np.ceil(label_size/2), np.ceil(label_size/2)])
    sumN = 0

    for i in range(label_size):
        for j in range(label_size):
            if np.linalg.norm(np.array([i, j]) - center) > rPos:
                label[0][i][j] = -1
                sumN += 1

    sumP = label_size ** 2 - sumN
    sumN = -sumN

    def callable(x):
        if x == -1:
            return 0.5 / sumN
        else:
            return 0.5 / sumP

    return label.apply_(callable)


class LogLoss(nn.Module):
    def __init__(self):
        super(LogLoss, self).__init__()
    
    def forward(self, x, y):
        b, c, h, w = x.shape
        cnt = b*c*h*w
        out = torch.log(1. + torch.exp(-x*y))
        out = torch.sum(out) / cnt
        return out