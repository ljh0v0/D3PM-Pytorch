## Adopted from https://github.com/rosinality/denoising-diffusion-pytorch with some minor changes.

import math

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import utils


def swish(input):
    return input * torch.sigmoid(input)


@torch.no_grad()
def variance_scaling_init_(tensor, scale=1, mode="fan_avg", distribution="uniform"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in

    elif mode == "fan_out":
        scale /= fan_out

    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)

        return tensor.normal_(0, std)

    else:
        bound = math.sqrt(3 * scale)

        return tensor.uniform_(-bound, bound)


def conv2d(
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        scale=1,
        mode="fan_avg",
):
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias
    )

    variance_scaling_init_(conv.weight, scale, mode=mode)

    if bias:
        nn.init.zeros_(conv.bias)

    return conv


def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    lin = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.init.zeros_(lin.bias)

    return lin


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)


class Upsample(nn.Sequential):
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)


class Downsample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.downsample = nn.Sequential(conv2d(channel, channel, 3, stride=2))


    def forward(self, input):
        input_pad = F.pad(input, [0, 1, 0, 1])

        return self.downsample(input_pad)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_dim, dropout):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channel, eps=1e-06)
        self.activation1 = Swish()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.time = nn.Sequential(Swish(), linear(time_dim, out_channel))

        self.norm2 = nn.GroupNorm(32, out_channel, eps=1e-06)
        self.activation2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        if in_channel != out_channel:
            # TODO
            # self.skip = conv2d(in_channel, out_channel, 1)
            self.skip = linear(in_channel, out_channel)

        else:
            self.skip = None

    def forward(self, input, time):
        batch = input.shape[0]

        out = self.conv1(self.activation1(self.norm1(input)))

        out = out + self.time(time).view(batch, -1, 1, 1)

        out = self.conv2(self.dropout(self.activation2(self.norm2(out))))

        if self.skip is not None:
            # TODO
            # input = self.skip(input)
            input = self.skip(input.permute(0, 2, 3, 1).contiguous())
            input = input.permute(0, 3, 1, 2)

        return out + input

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class SelfAttention(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, n_head=1):
        super().__init__()
        self.channels = channels
        self.num_heads = n_head

        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.
        Meant to be used like:
            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )
        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        half_dim = self.dim // 2
        self.inv_freq = torch.exp(torch.arange(half_dim, dtype=torch.float32) * (-math.log(10000) / (half_dim - 1)))

    def forward(self, input):
        shape = input.shape
        input = input.view(-1).to(torch.float32)
        sinusoid_in = torch.ger(input, self.inv_freq.to(input.device))
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class ResBlockWithAttention(nn.Module):
    def __init__(self, in_channel, out_channel, time_dim, dropout, attention_head=1, use_attention=False):
        super().__init__()

        self.resblocks = ResBlock(in_channel, out_channel, time_dim, dropout)

        if use_attention:
            self.attention = SelfAttention(out_channel, n_head=attention_head)

        else:
            self.attention = None

    def forward(self, input, time):
        out = self.resblocks(input, time)

        if self.attention is not None:
            out = self.attention(out)

        return out


def spatial_fold(input, fold):
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return (
        input.view(batch, channel, h_fold, fold, w_fold, fold)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(batch, -1, h_fold, w_fold)
    )


def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.view(batch, -1, unfold, unfold, height, width)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(batch, -1, h_unfold, w_unfold)
    )


class UNet(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            channel,
            channel_multiplier,
            n_res_blocks,
            attn_resolutions,
            num_heads=1,
            dropout=0,
            model_output=str,  # 'logits' or 'logistic_pars'
            num_pixel_vals=256,
            img_size=64
    ):
        super().__init__()

        self.model_output = model_output
        self.num_pixel_vals = num_pixel_vals
        self.out_channel = out_channel
        time_dim = channel * 4

        attn_strides = []
        for res in attn_resolutions:
            attn_strides.append(img_size // int(res))

        n_block = len(channel_multiplier)

        self.time = nn.Sequential(
            TimeEmbedding(channel),
            linear(channel, time_dim),
            Swish(),
            linear(time_dim, time_dim),
        )

        down_layers = [conv2d(in_channel, channel, 3, padding=1)]
        feat_channels = [channel]
        in_channel = channel
        for i in range(n_block):
            for _ in range(n_res_blocks):
                channel_mult = channel * channel_multiplier[i]

                down_layers.append(
                    ResBlockWithAttention(
                        in_channel,
                        channel_mult,
                        time_dim,
                        dropout,
                        attention_head=num_heads,
                        use_attention=2 ** i in attn_strides,
                    )
                )

                feat_channels.append(channel_mult)
                in_channel = channel_mult

            if i != n_block - 1:
                down_layers.append(Downsample(in_channel))
                feat_channels.append(in_channel)

        self.down = nn.ModuleList(down_layers)

        self.mid = nn.ModuleList(
            [
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    attention_head=num_heads,
                    use_attention=True,
                ),
                ResBlockWithAttention(
                    in_channel, in_channel, time_dim, dropout=dropout
                ),
            ]
        )

        up_layers = []
        for i in reversed(range(n_block)):
            for _ in range(n_res_blocks + 1):
                channel_mult = channel * channel_multiplier[i]

                up_layers.append(
                    ResBlockWithAttention(
                        in_channel + feat_channels.pop(),
                        channel_mult,
                        time_dim,
                        dropout=dropout,
                        attention_head=num_heads,
                        use_attention=2 ** i in attn_strides,
                    )
                )

                in_channel = channel_mult

            if i != 0:
                up_layers.append(Upsample(in_channel))

        self.up = nn.ModuleList(up_layers)

        if self.model_output == 'logistic_pars':
            # The output represents logits or the log scale and loc of a
            # logistic distribution.
            self.out = nn.Sequential(
                nn.GroupNorm(32, in_channel, eps=1e-06),
                Swish(),
                conv2d(in_channel, out_channel * 2, 3, padding=1, scale=1e-10),
            )
        else:
            self.out = nn.Sequential(
                nn.GroupNorm(32, in_channel, eps=1e-06),
                Swish(),
                conv2d(in_channel, out_channel * self.num_pixel_vals, 3, padding=1, scale=1e-10),
            )

    def forward(self, input, time):
        time_embed = self.time(time)

        feats = []
        #
        # out = spatial_fold(input, self.fold)
        batch_size, channels, height, width = input.shape
        input_onehot = F.one_hot(input.to(torch.int64), num_classes=self.num_pixel_vals)
        hid = input = utils.normalize_data(input)

        for layer in self.down:
            if isinstance(layer, ResBlockWithAttention):
                hid = layer(hid, time_embed)

            else:
                hid = layer(hid)

            feats.append(hid)

        for layer in self.mid:
            hid = layer(hid, time_embed)

        for layer in self.up:
            if isinstance(layer, ResBlockWithAttention):
                hid = layer(torch.cat((hid, feats.pop()), 1), time_embed)

            else:
                hid = layer(hid)

        out = self.out(hid)
        # out = spatial_unfold(out, self.fold)
        if self.model_output == 'logistic_pars':
            loc, log_scale = torch.chunk(out, 2, dim=1)
            out = torch.tanh(loc + input), log_scale
        else:
            out = torch.reshape(out, (batch_size, self.out_channel, self.num_pixel_vals, height, width))
            out = out.permute(0, 1, 3, 4, 2).contiguous()
            out = out + input_onehot

        return out
