# Gated channel transformation for visual recognition (CVPR2020)
import torch 
from torch import nn


class GCT(nn.Module):

    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = torch.ones((1, num_channels, 1, 1))
        self.gamma = torch.zeros((1, num_channels, 1, 1))
        self.beta = torch.zeros((1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):

        if self.mode == 'l2':
            embedding = (x.pow(2).sum(2, keepdim=True).sum(3, keepdim=True) +
                         self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / \
                (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)
        
        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum(2, keepdim=True).sum(
                3, keepdim=True) * self.alpha
            norm = self.gamma / \
                (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)
        else:
            print('Unknown mode!')

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


def main():
    attention_block = GCT(64)
    input = torch.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()