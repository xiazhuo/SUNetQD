import os
import numpy as np

import torch
from torch import nn
from torchinfo import summary


def bsp_torch(a, b):
    # let A = (A1|A2) and B = (B1|B2) return (A2|A1).(B1|B2)
    a1, a2 = torch.chunk(a, 2, dim=-1)
    a = torch.cat((a2, a1), dim=-1).to(torch.float32)
    b = b.to(torch.float32)
    return torch.matmul(a, b).to(torch.int64) % 2


class ResidualBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock2d, self).__init__()
        self.circ_pad = nn.CircularPad2d(2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()

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
    def __init__(self, in_channels, out_channels, use_attn=False):
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
    def __init__(self, in_channels, out_channels, use_attn=False):
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
            # self.res2 = ResidualBlock2d(out_channels, out_channels)
        else:
            self.attn = nn.Identity()
            # self.res2 = nn.Identity()

    def forward(self, x):
        x = self.res1(x)
        x = self.attn(x)
        # x = self.res2(x)
        return x


class DownSample(nn.Module):
    def __init__(self, n_channels, use_pad=False):
        super(DownSample, self).__init__()
        if use_pad:
            self.conv = nn.Sequential(nn.CircularPad2d(1),
                                      nn.Conv2d(n_channels, n_channels,
                                                kernel_size=3, stride=2))
        else:
            self.conv = nn.Conv2d(n_channels, n_channels,
                                  kernel_size=3, stride=1)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, use_pad=False):
        super(UpSample, self).__init__()
        if use_pad:
            self.conv = nn.Sequential(nn.CircularPad2d(1),
                                      nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, stride=2))
        else:
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
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=(2, 3, 3))
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act2 = nn.ReLU()

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
            # self.res2 = ResidualBlock3d(out_channels, out_channels)
        else:
            self.attn = nn.Identity()
            # self.res2 = nn.Identity()

    def forward(self, x):
        x = self.res1(x)
        x = self.attn(x)
        # x = self.res2(x)
        return x


class SAUNet(nn.Module):
    def __init__(self, channels_2d, channels_3d, n_measure=1, n_classes=4,
                 ch_mults=(1, 2, 2), use_attn=(False, False, True)):
        super(SAUNet, self).__init__()
        n_resolutions = len(ch_mults)
        out_channels = in_channels = channels_2d
        self.n_measure = n_measure
        self.n_classes = n_classes
        self.file_dir = os.path.join("low_decoder", "_".join(
            [str(k) for k in [channels_2d, channels_3d, n_measure, n_classes, ch_mults, use_attn]]))

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
        self.act = nn.ReLU()

        self.dim_expand3d = nn.Conv3d(
            in_channels=n_measure, out_channels=channels_3d, kernel_size=1)
        self.dim_shrink3d = nn.Conv3d(
            in_channels=channels_3d, out_channels=n_classes, kernel_size=1)
        self.middle3d = MiddleBlock3d(
            in_channels=channels_3d, out_channels=channels_3d)

    def forward(self, x):
        if x.dim() == 4:
            x.unsqueeze_(1)
        bz, n, _, d, d = x.shape
        x = x.reshape(-1, 2, d, d)
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

        x = self.dim_shrink2d(x)
        x = x.reshape(bz, n, 2, d, d)

        x = self.dim_expand3d(x)
        x = self.middle3d(x)
        x = self.dim_shrink3d(x)
        return x.squeeze()

    def load_exist_state(self, dir_path, err_factor, err_probs, device, file_name="model.pt"):
        file_path = os.path.join(
            dir_path, self.file_dir, str(err_factor)+"_"+str(err_probs), file_name)
        if os.path.exists(file_path):
            print("load model from", file_path)
            self.load_state_dict(torch.load(
                file_path, weights_only=True, map_location=device))
        else:
            print(f"\033[91mWarning: file path {file_path} not exist\033[0m")

    def save_state(self, dir_path, err_factor, err_probs, file_name="model.pt"):
        file_path = os.path.join(
            dir_path, self.file_dir, str(err_factor)+"_"+str(err_probs), file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)


class logic_Net(nn.Module):
    def __init__(self, channels_2d, channels_3d, n_measure=1, ch_mults=(1, 2, 2), use_attn=(False, False, True)):
        super(logic_Net, self).__init__()

        self.n_measure = n_measure
        self.n_classes = 1
        self.file_dir = os.path.join("high_decoder", "_".join(
            [str(k) for k in [channels_2d, channels_3d, n_measure, self.n_classes, ch_mults, use_attn]]))

        # self.syd_net = SAUNet(channels_2d, channels_3d,
        #                       n_measure,  1, ch_mults, use_attn)
        # self.rec_net = SAUNet(channels_2d, channels_3d,
        #                       n_measure,  1, ch_mults, use_attn)
        self.global_net = SAUNet(channels_2d, channels_3d,
                                 n_measure+1,  4, ch_mults, use_attn)
        self.final_net = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        )

    def forward(self, syd, rec):
        rec.unsqueeze_(1)
        features = torch.cat((syd, rec), dim=1)
        glob = self.global_net(features)
        # syd = self.syd_net(syd)
        # rec = self.rec_net(rec)
        # glob = rec + syd
        # glob = self.global_net(glob)
        y = self.final_net(glob)
        return y

    def load_exist_state(self, dir_path, err_factor, err_probs, device, file_name="model.pt"):
        file_path = os.path.join(
            dir_path, self.file_dir, str(err_factor)+"_"+str(err_probs), file_name)
        if os.path.exists(file_path):
            print("load model from", file_path)
            self.load_state_dict(torch.load(
                file_path, weights_only=True, map_location=device))
        else:
            print(f"\033[91mWarning: file path {file_path} not exist\033[0m")

    def save_state(self, dir_path, err_factor, err_probs, file_name="model.pt"):
        file_path = os.path.join(
            dir_path, self.file_dir, str(err_factor)+"_"+str(err_probs), file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(self.state_dict(), file_path)


if __name__ == "__main__":
    model = SAUNet(32, 32, ch_mults=(1, 2, 2),
                   use_attn=(True, True, True))
    summary(model, input_size=(256, 1, 2, 9, 9))
    # model = logic_Net(32, 32, ch_mults=(1, 2, 2),
    #                   use_attn=(True, True, True))
    # summary(model, input_size=[(256, 1, 2, 9, 9), (256, 2, 9, 9)])
