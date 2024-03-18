import numpy as np

import torch
from torch import nn, optim
from torchinfo import summary


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2d, self).__init__()
        self.circ_pad = nn.CircularPad2d(2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1),
                                          nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.circ_pad(x)
        h = self.act1(self.bn1(self.conv1(h)))
        h = self.act2(self.bn2(self.conv2(h)) + self.shortcut(x))
        return h


class AttentionBlock2d(nn.Module):
    def __init__(self, n_channels, num_heads=4):
        super(AttentionBlock2d, self).__init__()
        self.num_heads = num_heads
        assert n_channels % num_heads == 0

        self.qkv = nn.Conv2d(n_channels, n_channels * 3,
                             kernel_size=1, bias=False)
        self.proj = nn.Conv2d(n_channels, n_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)  # (B, 3C, H, W)
        q, k, v = qkv.reshape(B*self.num_heads, -1, H*W).chunk(3, dim=1)
        q = q.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        scale = 1. / np.sqrt(C//self.num_heads)
        attn = torch.bmm(q, k) * scale
        attn = attn.softmax(dim=-1)
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attn=True):
        super(DownBlock, self).__init__()
        self.res = ResidualBlock2d(in_channels, out_channels)
        if use_attn:
            self.attn = AttentionBlock2d(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        x = self.res(x)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attn=True):
        super(UpBlock, self).__init__()
        self.res = ResidualBlock2d(in_channels + out_channels, out_channels)
        if use_attn:
            self.attn = AttentionBlock2d(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x):
        x = self.res(x)
        x = self.attn(x)
        return x


class MiddleBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, use_attn=True):
        super(MiddleBlock2d, self).__init__()
        self.res1 = ResidualBlock2d(in_channels, out_channels)
        if use_attn:
            self.attn = AttentionBlock2d(out_channels)
            self.res2 = ResidualBlock2d(out_channels, out_channels)
        else:
            self.attn = nn.Identity()
            self.res2 = nn.Identity()

    def forward(self, x):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class DownSample(nn.Module):
    def __init__(self, n_channels):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3)

    def forward(self, x):
        return self.conv(x)


class ResidualBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3d, self).__init__()
        self.circ_pad = nn.CircularPad3d((2, 2, 2, 2, 1, 1))

        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=(2, 3, 3))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=(2, 3, 3))
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1),
                                          nn.BatchNorm3d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        h = self.circ_pad(x)
        h = self.act1(self.bn1(self.conv1(h)))
        h = self.act2(self.bn2(self.conv2(h)) + self.shortcut(x))
        return h


class AttentionBlock3d(nn.Module):
    def __init__(self, n_channels, num_heads=4):
        super(AttentionBlock3d, self).__init__()
        self.num_heads = num_heads
        assert n_channels % num_heads == 0

        self.qkv = nn.Conv3d(n_channels, n_channels * 3,
                             kernel_size=1, bias=False)
        self.proj = nn.Conv3d(n_channels, n_channels, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        qkv = self.qkv(x)  # (B, 3C, D, H, W)
        q, k, v = qkv.reshape(B*self.num_heads, -1, D*H*W).chunk(3, dim=1)
        q = q.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        scale = 1. / np.sqrt(C//self.num_heads)
        attn = torch.bmm(q, k) * scale
        attn = attn.softmax(dim=-1)
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, -1, D, H, W)
        h = self.proj(h)
        return h + x


class MiddleBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, use_attn=True):
        super(MiddleBlock3d, self).__init__()
        self.res1 = ResidualBlock3d(in_channels, out_channels)
        if use_attn:
            self.attn = AttentionBlock3d(out_channels)
            self.res2 = ResidualBlock3d(out_channels, out_channels)
        else:
            self.attn = nn.Identity()
            self.res2 = nn.Identity()

    def forward(self, x):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Net(nn.Module):
    def __init__(self, channels_2d, channels_3d, num_class=4,
                 ch_mults=(1, 2, 2), use_attn=(False, False, True)):
        super(Net, self).__init__()
        n_resolutions = len(ch_mults)
        out_channels = in_channels = channels_2d

        down = []
        for i in range(n_resolutions-1):
            out_channels = in_channels * ch_mults[i]
            down.append(
                DownBlock(in_channels, out_channels, use_attn=use_attn[i]))
            down.append(DownSample(out_channels))
            in_channels = out_channels
        self.down = nn.ModuleList(down)

        out_channels = in_channels * ch_mults[-1]
        self.middle2d = MiddleBlock2d(in_channels, out_channels, use_attn[-1])
        in_channels = out_channels

        up = []
        for i in range(n_resolutions-1, 0, -1):
            out_channels = in_channels // ch_mults[i]
            up.append(UpSample(in_channels, out_channels))
            in_channels = out_channels
            up.append(UpBlock(in_channels, out_channels,
                      use_attn=use_attn[i-1]))
        self.up = nn.ModuleList(up)

        self.dim_expand2d = nn.Conv2d(
            in_channels=2, out_channels=channels_2d, kernel_size=1)
        self.dim_shrink2d = nn.Conv2d(
            in_channels=channels_2d, out_channels=2, kernel_size=1)
        self.act = nn.SiLU()

        self.dim_expand3d = nn.Conv3d(
            in_channels=1, out_channels=channels_3d, kernel_size=1)
        self.dim_shrink3d = nn.Conv3d(
            in_channels=channels_3d, out_channels=num_class, kernel_size=1)
        self.middle3d = MiddleBlock3d(
            in_channels=channels_3d, out_channels=channels_3d)

    def forward(self, x):
        x = self.dim_expand2d(x)
        h = [x]
        for layer in self.down:
            if isinstance(layer, DownSample):
                h.append((x))
            x = layer(x)

        x = self.middle2d(x)

        for layer in self.up:
            if isinstance(layer, UpSample):
                x = layer(x)
            else:
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                x = layer(x)

        s = h.pop()
        x = self.dim_shrink2d(self.act(s + x))
        x = x.unsqueeze(1)

        x = self.dim_expand3d(x)
        s = x
        x = self.middle3d(x)
        x = self.dim_shrink3d(self.act(s + x))
        return x


if __name__ == "__main__":
    model = Net(6, 12, num_class=2)
    print(summary(model, input_size=(256, 2, 7, 7)))
