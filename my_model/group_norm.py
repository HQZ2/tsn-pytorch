import torch
import torch.nn as nn

__all__ = [
    'GroupNorm2d',
]

class GroupNorm2d(nn.Module):
    def __init__(self, channel_num, group_num = 32, eps = 1e-10, zero_gamma=False):
        super(GroupNorm2d,self).__init__()
        self.group_num = group_num
        if zero_gamma:
            self.gamma = nn.Parameter(torch.zeros(channel_num, 1, 1))
        else:
            self.gamma = nn.Parameter(torch.ones(channel_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(channel_num, 1, 1))
        self.eps = eps

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
        N, C, H, W = input.size()
        if C % self.group_num != 0:
            raise ValueError('expected channel num {} can be devided by group num {}'
                             .format(C,self.group_num))

    def forward(self, input):
        N, C, H, W = input.size()

        input = input.view(N, self.group_num, -1)

        mean = input.mean(dim = 2, keepdim = True)
        std = input.std(dim = 2, keepdim = True)

        input = (input - mean) / (std+self.eps)
        input = input.view(N, C, H, W)

        return input * self.gamma + self.beta