import torch
import torch.nn as nn
from torch.nn import functional as F

class Demo(nn.Module):
    def __init__(self):
        super(Demo, self).__init__()
        self.decoder_input = nn.Linear(1, 1)

    def forward(self,x):
        y = self.decoder_input(x)
        return y
