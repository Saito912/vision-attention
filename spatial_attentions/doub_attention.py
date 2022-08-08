# A2-Nets: Double Attention Networks (NIPS 2018)
import torch as torch
from torch import nn


class DoubleAtten(nn.Module):
    def __init__(self, in_c):
        super(DoubleAtten, self).__init__()
        self.in_c = in_c
        self.convA = nn.Conv2d(in_c, in_c, kernel_size=1)
        self.convB = nn.Conv2d(in_c, in_c, kernel_size=1)
        self.convV = nn.Conv2d(in_c, in_c, kernel_size=1)

    def forward(self, input):

        feature_maps = self.convA(input)
        atten_map = self.convB(input)
        b, _, h, w = feature_maps.shape

        feature_maps = feature_maps.view(b, 1, self.in_c, h*w)
        atten_map = atten_map.view(b, self.in_c, 1, h*w)
        global_descriptors = torch.mean(
            (feature_maps * torch.softmax(atten_map, dim=-1)), dim=-1)

        v = self.convV(input)
        atten_vectors = torch.softmax(
            v.view(b, self.in_c, h*w), dim=-1)
        out = torch.matmul(atten_vectors.permute(0, 2, 1),
                     global_descriptors).permute(0, 2, 1)

        return out.view(b, _, h, w)


def main():
    attention_block = DoubleAtten(64)
    input = torch.rand([4, 64, 32, 32])
    output = attention_block(input)
    print(input.size(), output.size())


if __name__ == '__main__':
    main()