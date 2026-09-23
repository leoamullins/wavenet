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

    def __init__(self, R, S, dilation, n_mels=None):
        super().__init__()
        self.conv = CausalConv1d(R, 2 * R, dilation=dilation)
        self.gate = GatedTanh()
        self.res = nn.Conv1d(R, R, kernel_size=1)
        self.skip = nn.Conv1d(R, S, kernel_size=1)
        self.cond = nn.Conv1d(n_mels, 2 * R, kernel_size=1) if n_mels else None
        with torch.no_grad():
            self.res.weight *= 0.1

    def forward(self, x, c=None):
        h = self.conv(x)
        if self.cond is not None:
            h = h + self.cond(c)
        z = self.gate(h)
        out = x + self.res(z)
        return out, self.skip(z)


class WaveNet(nn.Module):

    def __init__(
        self, R, S, n_layers=9, n_stacks=2, n_classes=256, n_mels=None, hop=None
    ):
        super().__init__()

        self.upsample = Upsample(hop=hop) if n_mels is not None else None

        # input
        self.embed = nn.Embedding(n_classes, R)
        self.input_conv = CausalConv1d(R, R, dilation=1)

        # dilations
        self.dilations = [2**i for i in range(n_layers)] * n_stacks
        self.blocks = nn.ModuleList(
            [ResidualBlock(R, S, dilation=d, n_mels=n_mels) for d in self.dilations]
        )

        # head
        self.head1 = nn.Conv1d(S, S, 1)
        self.head2 = nn.Conv1d(S, n_classes, 1)

    @property
    def rf(self):
        return 1 + 1 + sum(self.dilations)

    def forward(self, x, c=None):
        if (c is None) != (self.upsample is None):
            raise ValueError("pass c iff the model was built with n_mels")
        if c is not None:
            c = self.upsample(c)
            assert c.shape[-1] == x.shape[-1], (c.shape, x.shape)

        h = torch.transpose(self.embed(x), dim0=1, dim1=2)
        h = self.input_conv(h)

        skip_sum = 0
        for block in self.blocks:
            h, skip = block(h, c)
            skip_sum = skip_sum + skip
        out = self.head1(torch.relu(skip_sum))
        out = self.head2(torch.relu(out))
        return out


class Upsample(nn.Module):
    """stretching mel frames, by repeating frames hop times"""

    def __init__(self, hop):
        super().__init__()
        self.hop = hop

    def forward(self, c):
        # c is (B, N_MELS, F)
        return c.repeat_interleave(self.hop, dim=2)  # now it is (B, N_MELS, F * hop)
