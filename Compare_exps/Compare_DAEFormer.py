import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.autograd import Variable
from torch.utils import model_zoo
from torchvision import models
import argparse
from typing import Tuple
from metrics import f1_score, eval_metrics
from utils_comp_DAEF import JointTransform2D, ImageToImage2D, Image2D
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
import torch.utils.checkpoint as checkpoint

parser = argparse.ArgumentParser(description='MedT')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run(default: 400)')
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='batch size (default: 1)')
parser.add_argument('--learning_rate', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 0.001)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay (default: 1e-5)')
parser.add_argument('--train_dataset', default='D:/Data/Code_Dataset/MoNuSeg/Train_Folder', type=str)
parser.add_argument('--val_dataset', default='D:/Data/Code_Dataset/MoNuSeg/Val_Folder', type=str)
parser.add_argument('--save_freq', type=int, default=40)
parser.add_argument('--cuda', default="on", type=str, help='switch on/off cuda option (default: off)')
parser.add_argument('--load', default='default', type=str, help='load a pretrained model')
parser.add_argument('--save', default='default', type=str, help='save the model')
parser.add_argument('--direc', default="D:/Data/MedT_SAM/Compare/DAEFormer/MoNuSeg", type=str,help='directory to save')
parser.add_argument('--crop', type=int, default=None)
parser.add_argument('--device', default='cuda', type=str)

args = parser.parse_args()
direc = args.direc
imgsize = 128
imgchant = 3

tf_train = JointTransform2D(crop=None, p_flip=0.5, color_jitter_params=None, long_mask=True)
tf_val = JointTransform2D(crop=None, p_flip=0, color_jitter_params=None, long_mask=True)
train_dataset = ImageToImage2D(args.train_dataset, tf_train)
val_dataset = ImageToImage2D(args.val_dataset, tf_val)
predict_dataset = Image2D(args.val_dataset)
dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
valloader = DataLoader(val_dataset, 1, shuffle=True)
#segformer===================================

class EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class SelfAtten(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.head = head
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        if len(self.reduction_ratio) == 4:
            self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[3], reduction_ratio[3])
            self.sr1 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr2 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        elif len(self.reduction_ratio) == 3:
            self.sr0 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr1 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if len(self.reduction_ratio) == 4:
            tem0 = x[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
            tem1 = x[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem2 = x[:, 4704:5684, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem3 = x[:, 5684:6076, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))

        if len(self.reduction_ratio) == 3:
            tem0 = x[:, :1568, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem1 = x[:, 1568:2548, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem2 = x[:, 2548:2940, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, tem2], -2))

        return reduce_out


class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio  # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class LocalEnhance_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.local_pos = DWConv(dim)

        if reduction_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, reduction_ratio, reduction_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio > 1:
            p_x = x.clone().permute(0, 2, 1).reshape(B, C, H, W)
            sp_x = self.sr(p_x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(sp_x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)
        local_v = v.permute(0, 2, 1, 3).reshape(B, N, C)
        local_pos = self.local_pos(local_v).reshape(B, -1, self.head, C // self.head).permute(0, 2, 1, 3)
        x_atten = ((attn_score @ v) + local_pos).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)


class MixFFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.dwconv(self.fc1(x), H, W))
        out = self.fc2(ax)
        return out


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class MLP_FFN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MixD_FFN(nn.Module):
    def __init__(self, c1, c2, fuse_mode="add"):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1) if fuse_mode == "add" else nn.Linear(c2 * 2, c1)
        self.fuse_mode = fuse_mode

    def forward(self, x):
        ax = self.dwconv(self.fc1(x), H, W)
        fuse = self.act(ax + self.fc1(x)) if self.fuse_mode == "add" else self.act(torch.cat([ax, self.fc1(x)], 2))
        out = self.fc2(ax)
        return out


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class MixD_FFN(nn.Module):
    def __init__(self, c1, c2, fuse_mode="add"):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1) if fuse_mode == "add" else nn.Linear(c2 * 2, c1)
        self.fuse_mode = fuse_mode

    def forward(self, x):
        ax = self.dwconv(self.fc1(x), H, W)
        fuse = self.act(ax + self.fc1(x)) if self.fuse_mode == "add" else self.act(torch.cat([ax, self.fc1(x)], 2))
        out = self.fc2(ax)
        return out


class OverlapPatchEmbeddings(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, padding=1, in_ch=3, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride, padding)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        px = self.proj(x)
        _, _, H, W = px.shape
        fx = px.flatten(2).transpose(1, 2)
        nfx = self.norm(fx)
        return nfx, H, W


class TransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == "mix":
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx


class FuseTransformerBlock(nn.Module):
    def __init__(self, dim, head, reduction_ratio=1, fuse_mode="add"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAtten(dim, head, reduction_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixD_FFN(dim, int(dim * 4), fuse_mode)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        tx = x + self.attn(self.norm1(x), H, W)
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx


class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.proj(x)


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))


class MiT(nn.Module):
    def __init__(self, image_size, dims, layers, token_mlp="mix_skip"):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], dims[0], dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], dims[1], dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbeddings(
            image_size // 16, patch_sizes[3], strides[3], padding_sizes[3], dims[2], dims[3]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [TransformerBlock(dims[0], heads[0], reduction_ratios[0], token_mlp) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList(
            [TransformerBlock(dims[1], heads[1], reduction_ratios[1], token_mlp) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList(
            [TransformerBlock(dims[2], heads[2], reduction_ratios[2], token_mlp) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList(
            [TransformerBlock(dims[3], heads[3], reduction_ratios[3], token_mlp) for _ in range(layers[3])]
        )
        self.norm4 = nn.LayerNorm(dims[3])

        # self.head = nn.Linear(dims[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class FuseMiT(nn.Module):
    def __init__(self, image_size, dims, layers, fuse_mode="add"):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]
        reduction_ratios = [8, 4, 2, 1]
        heads = [1, 2, 5, 8]

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbeddings(image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, dims[0])
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], dims[0], dims[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], dims[1], dims[2]
        )
        self.patch_embed4 = OverlapPatchEmbeddings(
            image_size // 16, patch_sizes[3], strides[3], padding_sizes[3], dims[2], dims[3]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [FuseTransformerBlock(dims[0], heads[0], reduction_ratios[0], fuse_mode) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(dims[0])

        self.block2 = nn.ModuleList(
            [FuseTransformerBlock(dims[1], heads[1], reduction_ratios[1], fuse_mode) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(dims[1])

        self.block3 = nn.ModuleList(
            [FuseTransformerBlock(dims[2], heads[2], reduction_ratios[2], fuse_mode) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(dims[2])

        self.block4 = nn.ModuleList(
            [FuseTransformerBlock(dims[3], heads[3], reduction_ratios[3], fuse_mode) for _ in range(layers[3])]
        )
        self.norm4 = nn.LayerNorm(dims[3])

        # self.head = nn.Linear(dims[3], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


class Decoder(nn.Module):
    def __init__(self, dims, embed_dim, num_classes):
        super().__init__()

        self.linear_c1 = MLP(dims[0], embed_dim)
        self.linear_c2 = MLP(dims[1], embed_dim)
        self.linear_c3 = MLP(dims[2], embed_dim)
        self.linear_c4 = MLP(dims[3], embed_dim)

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim, 1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)

        self.conv_seg = nn.Conv2d(128, num_classes, 1)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = inputs
        n = c1.shape[0]
        c1f = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        c2f = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        c2f = F.interpolate(c2f, size=c1.shape[2:], mode="bilinear", align_corners=False)

        c3f = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        c3f = F.interpolate(c3f, size=c1.shape[2:], mode="bilinear", align_corners=False)

        c4f = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        c4f = F.interpolate(c4f, size=c1.shape[2:], mode="bilinear", align_corners=False)

        c = self.linear_fuse(torch.cat([c4f, c3f, c2f, c1f], dim=1))
        c = self.dropout(c)
        return self.linear_pred(c)


segformer_settings = {
    "B0": [[32, 64, 160, 256], [2, 2, 2, 2], 256],  # [channel dimensions, num encoder layers, embed dim]
    "B1": [[64, 128, 320, 512], [2, 2, 2, 2], 256],
    "B2": [[64, 128, 320, 512], [3, 4, 6, 3], 768],
    "B3": [[64, 128, 320, 512], [3, 4, 18, 3], 768],
    "B4": [[64, 128, 320, 512], [3, 8, 27, 3], 768],
    "B5": [[64, 128, 320, 512], [3, 6, 40, 3], 768],
}


class SegFormer(nn.Module):
    def __init__(self, model_name: str = "B0", num_classes: int = 19, image_size: int = 224) -> None:
        super().__init__()
        assert (
            model_name in segformer_settings.keys()
        ), f"SegFormer model name should be in {list(segformer_settings.keys())}"
        dims, layers, embed_dim = segformer_settings[model_name]

        self.backbone = MiT(image_size, dims, layers)
        self.decode_head = Decoder(dims, embed_dim, num_classes)

    def init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location="cpu"), strict=False)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        encoder_outs = self.backbone(x)
        return self.decode_head(encoder_outs)


#=====================================================
class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, height, width, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width

        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = context.transpose(1, 2) @ query  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value


class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, in_dim, key_dim, value_dim, height, width, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        self.attn = Cross_Attention(key_dim, value_dim, height, width, head_count=head_count)
        self.norm2 = nn.LayerNorm((in_dim * 2))
        if token_mlp == "mix":
            self.mlp = MixFFN((in_dim * 2), int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip((in_dim * 2), int(in_dim * 4))
        else:
            self.mlp = MLP_FFN((in_dim * 2), int(in_dim * 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)

        attn = self.attn(norm_1, norm_2)
        # attn = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(attn)

        # residual1 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x1)
        # residual2 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x2)
        residual = torch.cat([x1, x2], dim=2)
        tx = residual + attn
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W)
        return mx


class EfficientAttention(nn.Module):
    """
    input  -> x:[B, D, H, W]
    output ->   [B, D, H, W]

    in_channels:    int -> Embedding Dimension
    key_channels:   int -> Key Embedding Dimension,   Best: (in_channels)
    value_channels: int -> Value Embedding Dimension, Best: (in_channels or in_channels//2)
    head_count:     int -> It divides the embedding dimension by the head_count and process each part individually

    Conv2D # of Params:  ((k_h * k_w * C_in) + 1) * C_out)
    """

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        # dual attention structure, efficient attention first then transpose attention
        norm1 = self.norm1(x)
        norm1 = Rearrange("b (h w) d -> b d h w", h=H, w=W)(norm1)

        attn = self.attn(norm1)
        attn = Rearrange("b d h w -> b (h w) d")(attn)

        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W)

        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)

        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W)

        mx = add3 + mlp2
        return mx


# Encoder
class MiT(nn.Module):
    def __init__(self, image_size, in_dim, key_dim, value_dim, layers, head_count=1, token_mlp="mix_skip"):
        super().__init__()
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]
        padding_sizes = [3, 1, 1, 1]

        # patch_embed
        # layers = [2, 2, 2, 2] dims = [64, 128, 320, 512]
        self.patch_embed1 = OverlapPatchEmbeddings(
            image_size, patch_sizes[0], strides[0], padding_sizes[0], 3, in_dim[0]
        )
        self.patch_embed2 = OverlapPatchEmbeddings(
            image_size // 4, patch_sizes[1], strides[1], padding_sizes[1], in_dim[0], in_dim[1]
        )
        self.patch_embed3 = OverlapPatchEmbeddings(
            image_size // 8, patch_sizes[2], strides[2], padding_sizes[2], in_dim[1], in_dim[2]
        )

        # transformer encoder
        self.block1 = nn.ModuleList(
            [DualTransformerBlock(in_dim[0], key_dim[0], value_dim[0], head_count, token_mlp) for _ in range(layers[0])]
        )
        self.norm1 = nn.LayerNorm(in_dim[0])

        self.block2 = nn.ModuleList(
            [DualTransformerBlock(in_dim[1], key_dim[1], value_dim[1], head_count, token_mlp) for _ in range(layers[1])]
        )
        self.norm2 = nn.LayerNorm(in_dim[1])

        self.block3 = nn.ModuleList(
            [DualTransformerBlock(in_dim[2], key_dim[2], value_dim[2], head_count, token_mlp) for _ in range(layers[2])]
        )
        self.norm3 = nn.LayerNorm(in_dim[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs


# Decoder
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=2, p2=2, c=C // 4)
        x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(
            x, "b h w (p1 p2 c)-> b (h p1) (w p2) c", p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale**2)
        )
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(
        self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims * 2, key_dim, value_dim, input_size[0], input_size[1], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        if x2 is not None:  # skip connection exist
            b, h, w, c = x2.shape
            x2 = x2.view(b, -1, c)
            x1_expand = self.x1_linear(x1)
            cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2))
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


class DAEFormer(nn.Module):
    def __init__(self, num_classes=2, head_count=1, token_mlp_mode="mix_skip"):
        super().__init__()

        # Encoder
        dims, key_dim, value_dim, layers = [[128, 320, 512], [128, 320, 512], [128, 320, 512], [2, 2, 2]]
        self.backbone = MiT(
            image_size=224,
            in_dim=dims,
            key_dim=key_dim,
            value_dim=value_dim,
            layers=layers,
            head_count=head_count,
            token_mlp=token_mlp_mode,
        )

        # Decoder
        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [64, 128, 128, 128, 160],
            [320, 320, 320, 320, 256],
            [512, 512, 512, 512, 512],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        self.decoder_2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2),
            in_out_chan[2],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_1 = MyDecoderLayer(
            (d_base_feat_size * 4, d_base_feat_size * 4),
            in_out_chan[1],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
        )
        self.decoder_0 = MyDecoderLayer(
            (d_base_feat_size * 8, d_base_feat_size * 8),
            in_out_chan[0],
            head_count,
            token_mlp_mode,
            n_class=num_classes,
            is_last=True,
        )

    def forward(self, x):
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        output_enc = self.backbone(x)

        b, c, _, _ = output_enc[2].shape

        # ---------------Decoder-------------------------
        tmp_2 = self.decoder_2(output_enc[2].permute(0, 2, 3, 1).view(b, -1, c))
        tmp_1 = self.decoder_1(tmp_2, output_enc[1].permute(0, 2, 3, 1))
        tmp_0 = self.decoder_0(tmp_1, output_enc[0].permute(0, 2, 3, 1))

        return tmp_0


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


if __name__ == '__main__':
    device = torch.device("cuda")
    model = DAEFormer().to(device=device)
    model.load_state_dict(torch.load("D:/Data/MedT_SAM/Compare/DAEFormer/Covid19_text/_test_output/_weight/best_epoch95_best_iou0.8039_best_local.pth"))
    #model.load_state_dict(torch.load("./'result2_original_pyramid'final_model.pth"))
    # model.load_state_dict(torch.load("./'result2_Covid19_text/GLO-LO-Task04'/390/GLO-LO.pth"))
    # optimizer = torch.optim.Adam(list(model.parameters()), lr=args.learning_rate,weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0001)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    best_iou = 0
    num_classes = 2
    best_val_iou = 0
    patience = 12
    counter = 0
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    for epoch in range(args.epochs):

        epoch_running_loss = 0
        # F1 = 0
        # IoU = 0
        total_inter, total_union = 0, 0
        total_correct, total_label = 0, 0
        model.train()
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(dataloader):
            # X_batch = Variable(X_batch.to(device='cuda'))   #取值[0,1]
            # y_batch = Variable(y_batch.to(device='cuda'))   #取值0和1
            #print("========batch:{}".format(batch_idx))
            #print(X_batch[i].shape)
            X_batch = Variable(X_batch.to(device=device))       # [4,3,128,128]
            y_batch = Variable(y_batch.to(device=device))       # [4,128,128]

            output = model(X_batch)                  # [4,2,128,128]
            loss_ce = ce_loss(output, y_batch[:].long())
            loss_dice = dice_loss(output, y_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            # loss = F.cross_entropy(output, y_batch, weight=None, ignore_index=-100)
            #y_batch_i = y_batch_i.float()
            F1 = f1_score(output, y_batch)
            #metrics1 = Evaluation_Metrics(output, y_batch_i)
            seg_metrics = eval_metrics(output, y_batch, 2)
            total_correct += seg_metrics[0]
            total_label += seg_metrics[1]
            total_inter += seg_metrics[2]
            total_union += seg_metrics[3]
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            #dice = dice_coeff(output, y_batch_i)
            #iou = iouFun(output, y_batch_i)
            #IOU = jaccard_index(output, y_batch_i)

            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_running_loss += loss.item()
            # F1 += f1
            # IoU += iou
        # ===================log========================
        # print('epoch [{}/{}], loss:{:.4f}, F1:{}, IoU:{}'
        #       .format(epoch, args.epochs, epoch_running_loss / (batch_idx + 1), F1 / (batch_idx + 1), IoU / (batch_idx + 1)))
        print('epoch [{}/{}], loss:{:.4f}, F1:{}, Acc {:.4f} mIoU {:.4f}'
              .format(epoch, args.epochs, epoch_running_loss / (batch_idx + 1), F1, pixAcc, mIoU))

        # # 保存训练时候最优的模型参数
        # if (mIoU > best_iou):
        #     best_iou = mIoU
        #     fulldir = direc + "/_train_output"
        #     if not os.path.isdir(fulldir):
        #         os.makedirs(fulldir)
        #     torch.save(model.state_dict(),
        #                fulldir + '/best_epoch{}_best_iou{:.4f}'.format(epoch, best_iou) + "_best_local.pth")
        #     torch.save(model, direc + "AATC.pth")

        if epoch == 10:
            for param in model.parameters():
                param.requires_grad = True  # 打印参数个数

        # if (epoch % args.save_freq) == 0:  # 10的倍数 进行测试

        model.eval()
        with torch.no_grad():
            # val集进行验证
            for batch_idx, (X_batch_, y_batch_, *rest) in enumerate(valloader):
                if isinstance(rest[0][0], str):
                    image_filename = rest[0][0]
                else:
                    image_filename = '%s.png' % str(batch_idx + 1).zfill(3)

                X_batch_ = Variable(X_batch_.to(device=device))
                y_batch_ = Variable(y_batch_.to(device=device))
                y_out_ = model(X_batch_)

                # 计算val上的指标iou

                seg_metrics = eval_metrics(y_out_, y_batch_, 2)
                total_correct += seg_metrics[0]
                total_label += seg_metrics[1]
                total_inter += seg_metrics[2]
                total_union += seg_metrics[3]
                pixAcc_ = 1.0 * total_correct / (np.spacing(1) + total_label)
                val_IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
                val_mIoU = val_IoU.mean()

                tmp_ = y_out_.detach().cpu().numpy()
                tmp_[tmp_ >= 0.5] = 1
                tmp_[tmp_ < 0.5] = 0
                tmp_ = tmp_.astype(int)

                yHaT_ = tmp_

                del X_batch_, y_batch_, tmp_, y_out_

                yHaT_[yHaT_ == 1] = 255
                fulldir = direc + "/_test_output" + "/{}/".format(epoch)
                if not os.path.isdir(fulldir):
                    os.makedirs(fulldir)
                if (epoch % args.save_freq == 0):
                    cv2.imwrite(fulldir + image_filename, yHaT_[0, 1, :, :])
        # 保存训练时候最优的模型参数
        if (val_mIoU > best_val_iou)and(epoch>90):
            best_val_iou = val_mIoU
            fulldir = direc + "/_test_output" + "/_weight"
            if not os.path.isdir(fulldir):
                os.makedirs(fulldir)
            torch.save(model.state_dict(),
                        fulldir + '/best_epoch{}_best_iou{:.4f}'.format(epoch, best_val_iou) + "_best_local.pth")
            torch.save(model, direc + "AATC.pth")
            counter = 0
        else:
            counter+=1
        # 判断是否满足 Early Stopping 条件
        # if counter > patience:
        #     print("Validation metric did not improve for {} consecutive epochs. Training stopped.".format(patience))
        #     break