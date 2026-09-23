import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):

    def __init__(self, c_in, c_out, dilation, kernel_size=2):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class GatedTanh(nn.Module):

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)


class ResidualBlock(nn.Module):

    def __init__(self, R, S, dilation):
        super().__init__()
        self.conv = CausalConv1d(R, 2 * R, dilation=dilation)
        self.gate = GatedTanh()
        self.res = nn.Conv1d(R, R, kernel_size=1)
        self.skip = nn.Conv1d(R, S, kernel_size=1)
        with torch.no_grad():
            self.res.weight *= 0.1

    def forward(self, x):
        z = self.gate(self.conv(x))
        out = x + self.res(z)
        return out, self.skip(z)


class WaveNet(nn.Module):

    def __init__(self, R, S, n_layers=9, n_stacks=2, n_classes=256):
        super().__init__()

        # input
        self.embed = nn.Embedding(n_classes, R)
        self.input_conv = CausalConv1d(R, R, dilation=1)

        # dilations
        self.dilations = [2**i for i in range(n_layers)] * n_stacks
        self.blocks = nn.ModuleList(
            [ResidualBlock(R, S, dilation=d) for d in self.dilations]
        )

        # head
        self.head1 = nn.Conv1d(S, S, 1)
        self.head2 = nn.Conv1d(S, n_classes, 1)

    @property
    def rf(self):
        return 1 + 1 + sum(self.dilations)

    def forward(self, x):
        h = torch.transpose(self.embed(x), dim0=1, dim1=2)
        h = self.input_conv(h)

        skip_sum = 0
        for block in self.blocks:
            h, skip = block(h)
            skip_sum = skip_sum + skip
        out = self.head1(torch.relu(skip_sum))
        out = self.head2(torch.relu(out))
        return out
