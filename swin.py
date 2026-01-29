#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
"""
borrowed from the official swin implementation, with some modification.
search "prompt" for details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# 修改处：导入多层级MLP系统
from .multi_level_MLP import MultiLevelMLP, FeatureLossComputer

from .decoder_heads import SwinUperNetHead
# from warnings import deprecated

USE_STATIC = 2


#修改处 
class CNNFeatureExtractor(nn.Module):
    def __init__(self, embed_dim=768):
        super(CNNFeatureExtractor, self).__init__()
        # 示例架构：MobileNetV2风格的轻量卷积层组合
        # 第一层：标准卷积+批归一化+激活 (降低分辨率获取全局特征)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # 输出尺寸减半
        self.bn1 = nn.BatchNorm2d(32)
        # 深度可分离卷积块1：Depthwise卷积 + Pointwise卷积
        self.dw2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.pw2 = nn.Conv2d(32, 64, kernel_size=1)  # 通道数提升
        self.bn2 = nn.BatchNorm2d(64)
        # 深度可分离卷积块2
        self.dw3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
        self.pw3 = nn.Conv2d(64, embed_dim, kernel_size=1)  # 映射到embed_dim通道
        self.bn3 = nn.BatchNorm2d(embed_dim)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()
        
    def _init_weights(self):
        """
        权重初始化方法 - 遍历所有子模块并初始化
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                # 对于Conv2d层使用Kaiming He初始化
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                print(f"Initialized Conv2d: {name}")
                
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm层：权重初始化为1，偏置初始化为0
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print(f"Initialized BatchNorm2d: {name}")
            
    def forward(self, x):
        # x: 图像张量, shape [B,3,H,W]
        x = self.relu(self.bn1(self.conv1(x)))      # 第一层卷积
        x = self.relu(self.bn2(self.pw2(self.dw2(x))))  # 深度可分离卷积块1
        x = self.relu(self.bn3(self.pw3(self.dw3(x))))  # 深度可分离卷积块2，输出通道=embed_dim
        # 此时x维度: [B, embed_dim, H', W']，H'、W'为降低分辨率后的尺寸
        # 全局平均池化并取得3个token特征
        # 使用自适应平均池将特征图纵向划分为3块，分别池化
        # 如果希望简单地得到3个全局特征，也可直接全局池化后用多个全连接层得到3个向量
        feat_map = nn.functional.adaptive_avg_pool2d(x, output_size=(3, 1))  # [B, embed_dim, 3, 1]
        # 将3x1的空间维度展平成3个向量
        feat_map = feat_map.view(x.size(0), x.size(1), 3)   # [B, embed_dim, 3]
        B, D, H, W = feat_map.shape
        # 提取全局平均池化和全局最大池化特征作为两个prompt token
        f1 = F.adaptive_avg_pool2d(feat_map, (1,1)).view(B, -1)  # [B, D]
        f2 = F.adaptive_max_pool2d(feat_map, (1,1)).view(B, -1)  # [B, D]
        cnn_tokens = torch.stack([f1, f2], dim=1)  # 堆叠为 [B, 2, D]
        return cnn_tokens
#修改处

# 修改处：新增 Medium 和 High 等级的 CNN 提取模块，以及文本提取模块
class CNNFeatureExtractorMedium(nn.Module):
    def __init__(self, embed_dim=768):
        super(CNNFeatureExtractorMedium, self).__init__()
        # CNN中级特征提取器：结构类似 MobileNetV2 的轻量卷积层组合
        # 第一层：标准卷积+批归一化+ReLU（降低分辨率，提取基础特征）
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # 尺寸减半
        self.bn1 = nn.BatchNorm2d(32)
        # 深度可分离卷积块1：Depthwise卷积（按通道分组）+ Pointwise卷积（通道扩展）
        self.dw2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.pw2 = nn.Conv2d(32, 64, kernel_size=1)        # 提升通道数
        self.bn2 = nn.BatchNorm2d(64)
        # 深度可分离卷积块2：继续提取更高级别特征
        self.dw3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
        self.pw3 = nn.Conv2d(64, embed_dim, kernel_size=1)  # 映射到 embed_dim 维度
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()
        
    def _init_weights(self):
        """
        权重初始化方法 - 遍历所有子模块并初始化
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                # 对于Conv2d层使用Kaiming He初始化
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                print(f"Initialized Conv2d: {name}")
                
            elif isinstance(module, nn.BatchNorm2d):
                # BatchNorm层：权重初始化为1，偏置初始化为0
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print(f"Initialized BatchNorm2d: {name}")
                
    def forward(self, x):
        # 输入 x: 图像张量 [B, 3, H, W]
        x = self.relu(self.bn1(self.conv1(x)))            # 基础卷积层
        x = self.relu(self.bn2(self.pw2(self.dw2(x))))    # 深度可分离卷积块1
        x = self.relu(self.bn3(self.pw3(self.dw3(x))))    # 深度可分离卷积块2，输出通道=embed_dim
        # 特征图尺寸此时降低，形状 [B, embed_dim, H', W'] (H', W' ≈ 原图1/2)
        # 自适应平均池化获取3x3网格的区域特征，共9个token
        feat_map = nn.functional.adaptive_avg_pool2d(x, output_size=(3, 3))  # [B, embed_dim, 3, 3]
        # 将3x3的空间维展平成9个token向量
        feat_map = feat_map.view(x.size(0), x.size(1), 3 * 3)    # [B, embed_dim, 9]
        cnn_tokens = feat_map.permute(0, 2, 1)                  # [B, 9, embed_dim]
        return cnn_tokens

class CNNFeatureExtractorHigh(nn.Module):
    def __init__(self, embed_dim=768):
        super(CNNFeatureExtractorHigh, self).__init__()
        # CNN高级特征提取器：结构与Medium类似，用于提取更全局的特征
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dw2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.pw2 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dw3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64)
        self.pw3 = nn.Conv2d(64, embed_dim, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化 - 与Medium相同的策略"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                print(f"Initialized Conv2d: {name}")
                
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print(f"Initialized BatchNorm2d: {name}")
                
    def forward(self, x):
        # 输入 x: 图像张量 [B, 3, H, W]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.pw2(self.dw2(x))))
        x = self.relu(self.bn3(self.pw3(self.dw3(x))))
        # 修改：将自适应池化输出尺寸从 (2,1) 改为 (1,1)，确保每张图像仅提取1个全局特征token
        feat_map = nn.functional.adaptive_avg_pool2d(x, output_size=(1, 1))  # [B, embed_dim, 1, 1]
        feat_map = feat_map.view(x.size(0), x.size(1), 1)      # [B, embed_dim, 1]
        cnn_tokens = feat_map.permute(0, 2, 1)                # [B, 1, embed_dim]
        return cnn_tokens

class CNNTextExtractor(nn.Module):
    def __init__(self, embed_dim=768, out_segments=8, input_dim=768):
        super(CNNTextExtractor, self).__init__()
        self.out_segments = out_segments
        # 文本特征1D卷积提取模块：将文本序列转换为固定数量的token特征
        # 输入通道=文本嵌入维度768，使用逐层卷积提取局部和全局特征
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, stride=2, padding=1)  # 降低序列长度一半
        self.bn1 = nn.BatchNorm1d(256)
        self.dw2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, groups=256)
        self.pw2 = nn.Conv1d(256, 512, kernel_size=1)       # 提升通道至512
        self.bn2 = nn.BatchNorm1d(512)
        self.dw3 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, groups=512)
        self.pw3 = nn.Conv1d(512, embed_dim, kernel_size=1)  # 输出通道扩展到embed_dim
        self.bn3 = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()
        
    def _init_weights(self):
        """1D卷积的权重初始化"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                # 1D卷积使用相同的Kaiming初始化
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                print(f"Initialized Conv1d: {name}")
                
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
                print(f"Initialized BatchNorm1d: {name}")
    
    def forward(self, x):
        # 输入 x: 文本序列特征 [B, L, embed_dim] （B批次，L为文本长度）
        x = x.transpose(1, 2)                        # 转置为 [B, embed_dim, L] 以适配 Conv1d 输入格式
        x = self.relu(self.bn1(self.conv1(x)))       # 卷积层1：提取局部特征并将序列长度减半
        x = self.relu(self.bn2(self.pw2(self.dw2(x))))  # 深度可分离卷积块1
        x = self.relu(self.bn3(self.pw3(self.dw3(x))))  # 深度可分离卷积块2，输出通道=embed_dim
        # 修改处：使用自适应平均池化将序列划分为 out_segments 段
        feat_seq = nn.functional.adaptive_avg_pool1d(x, output_size=self.out_segments)  # [B, embed_dim, out_segments]
        text_tokens = feat_seq.permute(0, 2, 1)  # 修改处：转置为 [B, out_segments, embed_dim]
        #text_tokens = feat_seq.permute(0, 2, 1)     # [B, out_segments, embed_dim]
        return text_tokens
#修改处


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # 修改处：动态计算当前的prompt数量
        expected_patch_tokens = self.window_size[0] * self.window_size[1]
        current_num_prompts = N - expected_patch_tokens if N > expected_patch_tokens else 0

        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        # 修改处：根据实际prompt数量动态扩展relative_position_bias
        if current_num_prompts > 0:
            # expand relative_position_bias
            _C, _H, _W = relative_position_bias.shape
            # 为prompt tokens添加零偏置
            prompt_bias = torch.zeros(_C, current_num_prompts, _W, device=attn.device)
            relative_position_bias = torch.cat([prompt_bias, relative_position_bias], dim=1)
            
            # 为prompt与其他tokens的交互添加零偏置
            full_bias = torch.zeros(_C, _H + current_num_prompts, current_num_prompts, device=attn.device)
            relative_position_bias = torch.cat([full_bias, relative_position_bias], dim=-1)
   
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # 修改处：根据实际prompt数量调整mask
            nW = mask.shape[0]
            # 修改处：确保mask维度与输入匹配
            if mask.shape[1] != N or mask.shape[2] != N:
                new_mask = torch.zeros(nW, N, N, device=mask.device)
                if current_num_prompts > 0:
                    # prompt部分允许全连接注意力
                    new_mask[:, :current_num_prompts, :] = 0
                    new_mask[:, :, :current_num_prompts] = 0
                    # patch部分使用原始mask
                    patch_size = min(mask.shape[1], N - current_num_prompts)
                    new_mask[:, current_num_prompts:current_num_prompts+patch_size, 
                            current_num_prompts:current_num_prompts+patch_size] = mask[:, :patch_size, :patch_size]
                mask = new_mask
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        block_module=SwinTransformerBlock,
        # add two more parameters for prompt
        num_prompts=None,
        prompt_location=None,
        deep_prompt=None,
        use_instruct=True,
        d_cross=0,
        d_inter=0,
        moe_n_experts=0,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.deep_prompt = deep_prompt
        self.use_instruct = use_instruct
        self.num_prompts = num_prompts * USE_STATIC if deep_prompt else 0
        self.prompt_location = prompt_location
        self.d_cross = d_cross
        self.d_inter = d_inter
        # build blocks
        if num_prompts is not None:
            self.deep_prompt = deep_prompt
            self.num_prompts = num_prompts
            self.num_prompts = self.num_prompts * USE_STATIC
            if use_instruct:
                self.num_prompts += 1
            self.prompt_location = prompt_location
            self.blocks = nn.ModuleList(
                [
                    block_module(
                        self.num_prompts,
                        prompt_location,
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=(
                            drop_path[i] if isinstance(drop_path, list) else drop_path
                        ),  # noqa
                        norm_layer=norm_layer,
                        d_cross=d_cross,
                        d_inter=d_inter,
                        moe_n_experts=moe_n_experts,
                        is_last=(i == depth - 1),
                    )
                    for i in range(depth)
                ]
            )
            # self.prompt_norm = nn.ModuleList(
            #     [norm_layer(dim) for i in range(depth)]
            # )

            if self.deep_prompt and self.prompt_location != "prepend":
                raise ValueError(
                    "deep prompt mode for swin is only applicable to prepend"
                )
        else:
            self.blocks = nn.ModuleList(
                [
                    block_module(
                        # self.num_prompts,  # 确保传递正确的num_prompts
                        # prompt_location,
                        dim=dim,
                        input_resolution=input_resolution,
                        num_heads=num_heads,
                        window_size=window_size,
                        shift_size=0 if (i % 2 == 0) else window_size // 2,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop,
                        attn_drop=attn_drop,
                        drop_path=(
                            drop_path[i] if isinstance(drop_path, list) else drop_path
                        ),  # noqa
                        norm_layer=norm_layer,
                    )
                    for i in range(depth)
                ]
            )

        # patch merging layer
        if downsample is not None:
            if num_prompts is None:
                self.downsample = downsample(
                    input_resolution, dim=dim, norm_layer=norm_layer
                )
            else:
                # 修改处：使用正确计算的num_prompts
                actual_prompts = num_prompts  # 这里的num_prompts应该已经是正确的19
                print(f"DEBUG - BasicLayer downsample: num_prompts={num_prompts}, actual_prompts={actual_prompts}")
                if self.use_instruct:
                    self.downsample = downsample(
                        USE_STATIC * num_prompts + 1,
                        prompt_location,
                        deep_prompt,
                        input_resolution,
                        dim=dim,
                        norm_layer=norm_layer,
                    )
                else:
                    self.downsample = downsample(
                        USE_STATIC * num_prompts,
                        prompt_location,
                        deep_prompt,
                        input_resolution,
                        dim=dim,
                        norm_layer=norm_layer,
                    )
        else:
            self.downsample = None

    def forward(
        self,
        x,
        deep_prompt_embd=None,
        instruct_prompt_embd=None,
        use_attn_fuse: bool = False,
        use_mm_fuse: bool = False,
        cross_feature=None,
        all_prompt_experts=None,
        static_prompt=None,
        moe_scores=None,
    ):

        imp_losses = []
        # 修改处: 确保 x 为 3D 张量 (B, L, C)
        if x.dim() != 3:
            x = x.view(x.size(0), -1, x.size(-1))
        if self.deep_prompt and deep_prompt_embd is None and not use_mm_fuse:
            raise ValueError("need deep_prompt embddings")
        if self.use_instruct:
            assert (
                instruct_prompt_embd is not None
            ), "instruction tuning is used but no instruction prompt embeddings are provided"
        if not self.deep_prompt:
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
        else:
            B = x.shape[0]  # batchsize
            num_blocks = len(self.blocks)
            if use_mm_fuse or deep_prompt_embd.shape[1] != num_blocks:
                # first layer
                for i in range(num_blocks):
                    if i == 0:
                        x = self.blocks[i](x)

                    elif use_attn_fuse:
                        prompt_emb = deep_prompt_embd[
                            :, :, i, ...
                        ]  # all experts at this block

                        assert self.use_instruct
                        x = self.blocks[i].forward_attn_moe(
                            x,
                            prompt_emb,
                            instruct_prompt_embd,
                            moe_scores[:, i, :],  # perlayer routing assumed
                        )  # assume promptedSwinTransformerBlock
                    elif use_mm_fuse:
                        # **修复的MoPE路径**
                        # 1. 准备static prompt（单个block的prompt）
                        if static_prompt is not None:
                            # static_prompt shape: [num_blocks, prompt_len, dim]
                            # 选择当前block的static prompt并扩展到batch size
                            current_static_prompt = static_prompt[i].unsqueeze(0).expand(B, -1, -1)
                        else:
                            current_static_prompt = None
                    
                        # 2. 准备expert prompts（单个block的所有expert prompts）  
                        if all_prompt_experts is not None:
                            # all_prompt_experts shape: [B, num_experts, num_blocks, prompt_len, dim]
                            # 选择当前block的expert prompts
                            current_expert_prompts = all_prompt_experts[:, :, i, :, :]  # [B, num_experts, prompt_len, dim]
                        else:
                            current_expert_prompts = None

                        # 3. 调用block的MoPE forward
                        if hasattr(self.blocks[i], 'forward_mm_moe'):
                            block_output = self.blocks[i].forward_mm_moe(
                                x,
                                current_static_prompt,
                                current_expert_prompts,
                            instruct_prompt_embd,
                            cross_feature,
                            )
                        
                            # 4. 处理返回值（tuple或tensor）
                            if isinstance(block_output, tuple):
                                x, imp_loss = block_output
                                imp_losses.append(imp_loss)
                            else:
                                x = block_output
                        else:
                            # 如果block没有forward_mm_moe方法，使用普通forward
                            x = self.blocks[i](x)
                    else:
                        # 普通深度prompt模式
                        prompt_emb = deep_prompt_embd[:, i, ...]
                        if self.use_instruct:
                            prompt_emb = torch.cat((prompt_emb, instruct_prompt_embd), dim=1)

                        x = torch.cat((prompt_emb, x[:, self.num_prompts :, :]), dim=1)
                        x = self.blocks[i](x)

            else:
                # other layers
                for i in range(num_blocks):
                    prompt_emb = deep_prompt_embd[:, i, ...]
                    # prompt_emb = deep_prompt_embd[i].expand(B, -1, -1)
                    if instruct_prompt_embd is not None:
                        prompt_emb = torch.cat(
                            (prompt_emb, instruct_prompt_embd), dim=1
                        )
                    # prompt_emb = self.prompt_norm[i](prompt_emb)
                    x = torch.cat((prompt_emb, x[:, self.num_prompts :, :]), dim=1)
                    x = self.blocks[i](x)

        if self.downsample is not None:
            x = self.downsample(x)
        # if the imp_loss is not empty, then we need to return i
        if len(imp_losses) > 0:
            return x, imp_losses
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = (
            Ho
            * Wo
            * self.embed_dim
            * self.in_chans
            * (self.patch_size[0] * self.patch_size[1])
        )
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(nn.Module):
    r"""Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=to_2tuple(img_size),
            patch_size=to_2tuple(patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                use_instruct=False,
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        #修改处
        # CNN增强模块: 添加CLS token和多级CNN特征提取器
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=0.02)
        # 修改处: 添加文本特征投影层，将 768 投影到 embed_dim
        self.text_proj = nn.Linear(768, embed_dim)
        #如果要在low级CNN中使用MoPE，则需要解除下面的注释
        #self.cnn_extractor = CNNFeatureExtractor(embed_dim=embed_dim)
        self.cnn_extractor_medium = CNNFeatureExtractorMedium(embed_dim=embed_dim)
        self.cnn_extractor_high   = CNNFeatureExtractorHigh(embed_dim=embed_dim)
        # 文本提取器分别用于 Medium 和 High （输出8个和2个token）
        self.cnn_extractor_text_med  = CNNTextExtractor(embed_dim=embed_dim, out_segments=8, input_dim=768)
        self.cnn_extractor_text_high = CNNTextExtractor(embed_dim=embed_dim, out_segments=1, input_dim=768)
        #修改处

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        f = self.norm(x)  # B L C
        cls_ = self.avgpool(f.transpose(1, 2))  # B C 1
        cls_ = torch.flatten(cls_, 1)
        return cls_, f

    def forward(self, x):
        cls_, f = self.forward_features(x)
        logit = self.head(cls_)
        return logit

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += (
            self.num_features
            * self.patches_resolution[0]
            * self.patches_resolution[1]
            // (2**self.num_layers)
        )
        flops += self.num_features * self.num_classes
        return flops


# ------------prompt version-----------------

import torchvision as tv
import math

from functools import reduce
from operator import mul
from torch.nn import Conv2d, Dropout

from timm.models.layers import to_2tuple


class PromptedSwinTransformer(SwinTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        prompt_length=10,
        prompt_medium_len=9,
        prompt_high_len=2,
        prompt_type: str = "vpt",
        moe_n_experts: int = 8,
        use_static_prompt=False,
        use_instruct=True,
        prompt_init="uniform",
        d_cross=0,
        d_inter=0,
        **kwargs,
    ):
        super(PromptedSwinTransformer, self).__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dim,
            depths,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            norm_layer,
            ape,
            patch_norm,
            use_checkpoint,
            **kwargs,
        )
        
        # 修改处：添加调试信息，检查prompt长度配置
        # print(f"DEBUG - PromptedSwinTransformer __init__:")
        # print(f"  prompt_length: {prompt_length}")
        # print(f"  prompt_medium_len: {prompt_medium_len}")
        # print(f"  prompt_high_len: {prompt_high_len}")
        # print(f"  use_instruct: {use_instruct}")
        # 修改处：初始化多层级MLP系统（替换原有的MoPE）
        d_moe_low = d_inter if d_inter > 0 else embed_dim  # 确保d_moe_low有合理的值
        # 如果已经存在投影层，从投影层获取维度
        if hasattr(self, 'low_router_self_proj'):
            d_moe_low = self.low_router_self_proj.out_features

        print(f"DEBUG - d_moe_low: {d_moe_low}, embed_dim: {embed_dim}")  # 调试信息
        # 修改处：添加多层级MLP系统
        self.use_hierarchical_mlp = True
        self.multi_level_mlp = MultiLevelMLP(
            embed_dim=embed_dim, 
            n_experts=moe_n_experts,
            d_moe_low=d_moe_low
        )
        self.feature_loss_computer = FeatureLossComputer(embed_dim=embed_dim)
        global USE_STATIC  # TODO Sep 30: refactor
        USE_STATIC = 2 if use_static_prompt else 1
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # Define key parameters for MoPE
        self.moe_n_experts = moe_n_experts            # register number of experts early
        self.num_tokens = prompt_length               # low-level prompt length from args
        # 修改处：计算 Prompt 占位符总数（移除低级CNN token占位）
        if not use_instruct:
            prompt_medium_len = 0
            prompt_high_len = 0
        # **修改**：根据是否使用静态Prompt/多模态，计算总的 Prompt 前缀长度
        if use_static_prompt:
            if use_instruct:
                # Static模式多模态：静态/动态两部分Prompt合计长度取半（图像CLS额外占位已计入，文本CLS稍后添加）
                extra_tokens = (1 + 2 * prompt_length + prompt_medium_len + prompt_high_len) // 2
            else:
                # Static模式单模态：仅考虑静态和动态部分共享图像CLS，不添加额外前缀
                extra_tokens = (0 + 2 * prompt_length) // 2
        else:
            if use_instruct:
                # Dynamic模式多模态：包含图像CLS和文本CLS各1个
                extra_tokens = 1 + prompt_medium_len + prompt_high_len + 1
            else:
                # Dynamic模式单模态：无额外前缀CLS（图像CLS嵌入在 prompt_low 第一位）
                extra_tokens = 0
        num_tokens = prompt_length + extra_tokens
        # Initialize prompt hyperparameters
        # snli
        prompt_medium_len = 9
        prompt_high_len = 2
        # extra
        # prompt_medium_len = 17
        # prompt_high_len = 4
        self.prompt_length = prompt_length
        self.prompt_medium_len = prompt_medium_len
        self.prompt_high_len = prompt_high_len
        # 修改处：添加文本特征投影层，将BERT的768维投影到Swin的embed_dim
        self.text_proj = nn.Linear(768, embed_dim)  # 修改处：确保文本特征维度匹配
        # 修改：定义高级 MoPE CNN 输出 token 数和 Prompt 长度，便于后续调整
        self.high_cnn_img_token_count = 1    # 高级MoPE每张图像由CNN提取1个token特征
        self.high_cnn_txt_token_count = 1    # 高级MoPE每段文本由CNN提取1个token特征
        # 定义中级 MoPE CNN 输出 token 数和 Prompt 长度
        self.medium_cnn_img_token_count = 9   # 中级MoPE每张图像由CNN提取9个token特征
        self.medium_cnn_txt_token_count = 8   # 中级MoPE每段文本由CNN提取8个token特征
        #修改处
        #如果要在low级CNN中使用MoPE，则需要解除下面的注释
        #self.cnn_extractor = CNNFeatureExtractor(embed_dim=embed_dim)
        # 修改处：初始化 Medium-MoPE 和 High-MoPE 的 CNN 提取模块及路由/专家参数
        self.cnn_extractor_medium = CNNFeatureExtractorMedium(embed_dim=embed_dim)
        self.cnn_extractor_high   = CNNFeatureExtractorHigh(embed_dim=embed_dim)
        # 文本提取器分别用于 Medium 和 High （输出8个和1个token）
        self.cnn_extractor_text_med  = CNNTextExtractor(embed_dim=embed_dim, out_segments=8)
        self.cnn_extractor_text_high = CNNTextExtractor(embed_dim=embed_dim, out_segments=1)
        # 为 Medium MoPE 初始化路由器投影层和专家参数
        d_moe_med = embed_dim
        self.moe_n_experts_med = self.moe_n_experts
        #self.medium_router_self_proj  = nn.Linear(embed_dim, d_moe_med)
        #self.medium_router_cross_proj = nn.Linear(embed_dim, d_moe_med)
        # 修改处：路由embedding维度应该匹配CNN输出维度
        self.medium_key_embed = nn.Parameter(torch.zeros(embed_dim, self.moe_n_experts_med), requires_grad=False)
        nn.init.orthogonal_(self.medium_key_embed)
        self.medium_key_embed.requires_grad = False
        # 每个 Medium 专家 Prompt 长度 = prompt_medium_len
        self.prompt_experts_medium = nn.Parameter(torch.randn(self.moe_n_experts_med, self.prompt_medium_len, embed_dim))
        nn.init.trunc_normal_(self.prompt_experts_medium, std=0.02)
        # 为 High MoPE 初始化路由器投影层和专家参数
        d_moe_high = embed_dim
        #self.high_router_self_proj  = nn.Linear(embed_dim, d_moe_high)
        #self.high_router_cross_proj = nn.Linear(embed_dim, d_moe_high)
        self.moe_n_experts_high = self.moe_n_experts
        # 修改处：路由embedding维度应该匹配CNN输出维度
        self.high_key_embed = nn.Parameter(torch.zeros(embed_dim, self.moe_n_experts_high), requires_grad=False)
        nn.init.orthogonal_(self.high_key_embed)
        self.high_key_embed.requires_grad = False
        # 每个 High 专家 Prompt 长度 = prompt_high_len (修改前为4，现为2)
        self.prompt_experts_high = nn.Parameter(torch.randn(self.moe_n_experts_high, self.prompt_high_len, embed_dim))
        nn.init.trunc_normal_(self.prompt_experts_high, std=0.02)
        # 修改处：初始化 Low MoPE 路由器投影层和专家 Prompt 参数
        d_moe_low = embed_dim  # 路由空间维度（设为embed_dim）
        self.low_router_self_proj  = nn.Linear(embed_dim, d_moe_low)
        self.low_router_cross_proj = nn.Linear(768, d_moe_low)
        self.moe_n_experts_low = self.moe_n_experts
        self.low_key_embed = nn.Parameter(torch.zeros(d_moe_low * 2, self.moe_n_experts_low), requires_grad=False)
        nn.init.orthogonal_(self.low_key_embed); self.low_key_embed.requires_grad = False
        # 每个 Low 专家 Prompt 长度 = prompt_length
        self.prompt_experts_low = nn.Parameter(torch.randn(self.moe_n_experts_low, self.prompt_length, embed_dim))
        nn.init.trunc_normal_(self.prompt_experts_low, std=0.02)
        # 全局池化用于 Medium/High Prompt 路由特征
        self.mm_prompt_global_pooler = nn.AdaptiveAvgPool1d(1)
        # 修改处
        self.prompt_dropout = Dropout(0.1)
        self.prompt_type = prompt_type
        self.depths = depths
        self.use_instruct = use_instruct
        # if project the prompt embeddings
        # if self.prompt_config.PROJECT > -1:
        self.prompt_proj = nn.Identity()

        # build layers
        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        # Calculate base number of prompt tokens (excluding instruct token)
        # 修改处：正确计算prompt tokens总数
        if use_instruct:
            num_tokens = 1 + 1 + prompt_length + prompt_medium_len + prompt_high_len  # img_cls + txt_cls + prompts
        else:
            num_tokens = 1 + prompt_length  # img_cls + prompts
        print(f"DEBUG - PromptedSwinTransformer init: calculated num_tokens = {num_tokens}")
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    self.patches_resolution[0] // (2**i_layer),
                    self.patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                block_module=PromptedSwinTransformerBlock,
                downsample=(
                    PromptedPatchMerging if (i_layer < self.num_layers - 1) else None
                ),
                use_checkpoint=use_checkpoint,
                num_prompts=num_tokens,
                prompt_location="prepend",
                deep_prompt=True,
                use_instruct=use_instruct,  #!HARDCODED: now, always use instruct
                d_cross=d_cross,
                d_inter=d_inter,
                moe_n_experts=moe_n_experts,
            )
            self.layers.append(layer)


        if True:
            # elif True: 
            self.moe_n_experts = moe_n_experts
            self.moe_top_k = 1
        val = math.sqrt(
            6.0 / float(3 * reduce(mul, patch_size, 1) + embed_dim)
        )  # noqa
        # for "prepend"
        self.extra_prompt_for_static = (
            1  # always add one for static prompt, even if not used
        )
        self.num_prompts = USE_STATIC * num_tokens
        if self.use_instruct:
            self.num_prompts += 1
        self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.num_prompts, embed_dim))
        # Set max text token length to ensure total sequence ≤ 512
        self.max_txt_len = 512 - self.num_prompts
        #! since now by-default inited with mope, the expert-0 is always the static prompt.
        #! so we always have an additional static prompt, and the total #prompt would be moe_n_expert + 1 per block
        self.deep_prompt_embeddings_0 = nn.Parameter(
            torch.zeros(
                self.moe_n_experts + self.extra_prompt_for_static,
                # depths[0] - 1,
                depths[0],
                num_tokens,
                embed_dim,
            )
        )

        self.deep_prompt_embeddings_1 = nn.Parameter(
            torch.zeros(
                self.moe_n_experts + self.extra_prompt_for_static,
                depths[1],
                num_tokens,
                embed_dim * 2,
            )
        )

        self.deep_prompt_embeddings_2 = nn.Parameter(
            torch.zeros(
                self.moe_n_experts + self.extra_prompt_for_static,
                depths[2],
                num_tokens,
                embed_dim * 4,
            )
        )

        self.deep_prompt_embeddings_3 = nn.Parameter(
            torch.zeros(
                self.moe_n_experts + self.extra_prompt_for_static,
                depths[3],
                num_tokens,
                embed_dim * 8,
            )
        )
        self._init_prompt(method=prompt_init, val=val)
        # additional projection from text (instruction prompt)
        if True:  # set false to avoid extra param for ablation
            self.prompt_proj_act = nn.GELU()
            self.prompt_proj_0 = nn.Linear(384, embed_dim)
            self.prompt_proj_1 = nn.Linear(384, embed_dim * 2)
            self.prompt_proj_2 = nn.Linear(384, embed_dim * 4)
            self.prompt_proj_3 = nn.Linear(384, embed_dim * 8)

            #!HARDCODED Sep 21: these are the proj for multimodal moe
            #!HARDCODED Oct 31:  default to route per swin layer (4 layer)
            self.mm_prompt_instruction_proj = nn.Sequential(
                nn.Linear(768, 384), nn.BatchNorm1d(384), nn.GELU()
            )


    def _init_prompt(self, method: str = "uniform", val=None):
        """
        How static prompt and prompt expert embeddings are initialized
        """
        if method == "uniform":
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            nn.init.uniform_(self.deep_prompt_embeddings_0.data, -val, val)
            nn.init.uniform_(self.deep_prompt_embeddings_1.data, -val, val)
            nn.init.uniform_(self.deep_prompt_embeddings_2.data, -val, val)
            nn.init.uniform_(self.deep_prompt_embeddings_3.data, -val, val)
        elif method == "normal":
            nn.init.trunc_normal_(self.prompt_embeddings.data, std=1)
            nn.init.trunc_normal_(self.deep_prompt_embeddings_0.data, std=1)
            nn.init.trunc_normal_(self.deep_prompt_embeddings_1.data, std=1)
            nn.init.trunc_normal_(self.deep_prompt_embeddings_2.data, std=1)
            nn.init.trunc_normal_(self.deep_prompt_embeddings_3.data, std=1)
        elif method == "othorgonal":
            nn.init.orthogonal_(self.prompt_embeddings.data)
            # othor for each block
            for i in range(self.depths[0] - 1):
                nn.init.orthogonal_(self.deep_prompt_embeddings_0[:, i, ...].data)
            for i in range(self.depths[1]):
                nn.init.orthogonal_(self.deep_prompt_embeddings_1[:, i, ...].data)
            for i in range(self.depths[2]):
                nn.init.orthogonal_(self.deep_prompt_embeddings_2[:, i, ...].data)
            for i in range(self.depths[3]):
                nn.init.orthogonal_(self.deep_prompt_embeddings_3[:, i, ...].data)

    # #修改处
    def incorporate_prompt(self, x, cross_feature=None):
         # 在方法开始处添加参数量统计
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        B = x.shape[0]
        patch_tokens = self.patch_embed(x)
        if self.ape:
            patch_tokens = patch_tokens + self.absolute_pos_embed
        patch_tokens = self.pos_drop(patch_tokens)

        # 计算低级 MoPE 路由分数
        low_img_feat = self.mm_prompt_global_pooler(patch_tokens.transpose(1, 2)).squeeze(-1)
    
        if cross_feature is not None:
            if cross_feature.dim() == 2:
                cross_feature = cross_feature.unsqueeze(1)
            low_txt_feat = self.mm_prompt_global_pooler(cross_feature.transpose(1, 2)).squeeze(-1)
        else:
            low_txt_feat = torch.zeros(B, 768, device=low_img_feat.device)
    
        # 低级MoPE路由计算
        moe_self_low = self.low_router_self_proj(low_img_feat)
        moe_cross_low = self.low_router_cross_proj(low_txt_feat)
        # 确保batch size匹配
        min_batch = min(moe_self_low.shape[0], moe_cross_low.shape[0])
        if moe_self_low.shape[0] != moe_cross_low.shape[0]:
            print(f"WARNING: Batch size mismatch detected! Image: {moe_self_low.shape[0]}, Text: {moe_cross_low.shape[0]}")
            print(f"Aligning to minimum batch size: {min_batch}")
            moe_self_low = moe_self_low[:min_batch]
            moe_cross_low = moe_cross_low[:min_batch]
            # 同时需要更新B以保持后续一致性
            B = min_batch
        low_feat = torch.cat([moe_self_low, moe_cross_low], dim=1)

        if cross_feature is not None:
            # 中级和高级特征提取
            medium_img_tokens = self.cnn_extractor_medium(x)
            if medium_img_tokens.dim() == 4:
                B_img = medium_img_tokens.shape[0]
                medium_img_tokens = medium_img_tokens.view(B_img, -1, medium_img_tokens.shape[-1])
        
            medium_txt_tokens = self.cnn_extractor_text_med(cross_feature)
            high_img_tokens = self.cnn_extractor_high(x)
            if high_img_tokens.dim() == 4:
                B_img = high_img_tokens.shape[0]
                high_img_tokens = high_img_tokens.view(B_img, -1, high_img_tokens.shape[-1])
        
            high_txt_tokens = self.cnn_extractor_text_high(cross_feature)
            
            medium_combined = torch.cat([medium_img_tokens, medium_txt_tokens], dim=1)
            med_feat = self.mm_prompt_global_pooler(medium_combined.transpose(1, 2)).squeeze(-1)
            
            high_combined = torch.cat([high_img_tokens, high_txt_tokens], dim=1)
            high_feat = self.mm_prompt_global_pooler(high_combined.transpose(1, 2)).squeeze(-1)


            # 修改处：通过多层级MLP系统获取路由分数
            mlp_outputs = self.multi_level_mlp(low_feat, med_feat, high_feat)
    
            # 修改处：使用路由分数加权对应的专家prompt
            # 低层级prompt（长度=prompt_length，例如10个token）
            low_scores = mlp_outputs['low_cross']  # [B, n_experts]
            all_expert_prompts_low = self.prompt_experts_low.unsqueeze(0).expand(B, -1, -1, -1)  # [B, n_experts, prompt_length, embed_dim]
            prompt_low = torch.einsum('bk, bknh -> bnh', low_scores, all_expert_prompts_low)  # [B, prompt_length, embed_dim]
    
            # 中层级prompt（长度=prompt_medium_len，例如9个token）
            med_scores = mlp_outputs['med_cross']  # [B, n_experts]
            all_expert_prompts_med = self.prompt_experts_medium.unsqueeze(0).expand(B, -1, -1, -1)  # [B, n_experts, prompt_medium_len, embed_dim]
            prompt_medium = torch.einsum('bk, bknh -> bnh', med_scores, all_expert_prompts_med)  # [B, prompt_medium_len, embed_dim]
    
            # 高层级prompt（长度=prompt_high_len，例如2个token）
            high_scores = mlp_outputs['high_cross']  # [B, n_experts]
            all_expert_prompts_high = self.prompt_experts_high.unsqueeze(0).expand(B, -1, -1, -1)  # [B, n_experts, prompt_high_len, embed_dim]
            prompt_high = torch.einsum('bk, bknh -> bnh', high_scores, all_expert_prompts_high)  # [B, prompt_high_len, embed_dim]


            # 拼接所有prompt部分 - 确保顺序和数量固定
            cls_img = self.cls_token.expand(B, 1, -1)
            cls_txt = self.text_proj(cross_feature[:, 0, :]).unsqueeze(1)

       # prefix_tokens = torch.cat([cls_img, cls_txt, prompt_low, prompt_medium, prompt_high], dim=1)
            prefix_tokens = torch.cat([cls_img, cls_txt, prompt_medium, prompt_high], dim=1)
        
            # 修改处：验证最终结果
            # expected_total = 1 + 1 + prompt_low.shape[1] + prompt_medium.shape[1] + prompt_high.shape[1]
            expected_total = 1 + 1 + prompt_medium.shape[1] + prompt_high.shape[1]
            actual_total = prefix_tokens.shape[1]

        
        else:
            cls_img = self.cls_token.expand(B, 1, -1)
            prefix_tokens = torch.cat([cls_img, prompt_low], dim=1)
            # print(f"  Final prefix_tokens shape (no text): {prefix_tokens.shape}")

        # 修改处：存储mlp_outputs用于损失计算
        # 修改处：存储MLP输出用于损失计算
        self.current_mlp_outputs = mlp_outputs
        self.current_multi_level_mlp = self.multi_level_mlp
        return prefix_tokens

    def get_patch_embeddings(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            # first set all to eval and set the prompt to train later
            for module in self.children():
                module.train(False)
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        B = x.shape[0]
        # 每个 stage（layer）前分别调用 incorporate_prompt
        for i, layer in enumerate(self.layers):
            # 如果使用多模态指导，则需要传递 text 特征
            if hasattr(self, 'use_instruct') and self.use_instruct:
                x = self.incorporate_prompt(x, cross_feature=self.current_text_feat)
            else:
                x = self.incorporate_prompt(x, cross_feature=self.current_text_feat)
            # 静态深度 Prompt 嵌入（只取第0位）
            deep_prompt_embd = self.deep_prompt_embeddings_[i][0].expand(B, -1, -1, -1)
            deep_prompt_embd = self.prompt_dropout(deep_prompt_embd)
            x = layer(x, deep_prompt_embd)
        f = self.norm(x)
        cls_ = self.avgpool(f.transpose(1, 2)).flatten(1)
        return cls_, f

    def forward_features_instruct(self, x, y, return_internal=False):
        """
        x: image
        y: text features
        """
        B = x.shape[0]
        x = self.incorporate_prompt(x, cross_feature=y)

        # if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:
        # forward instruct is forward with static prompt only
        y0 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_0(y))).unsqueeze(
            1
        )
        y1 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_1(y))).unsqueeze(
            1
        )
        y2 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_2(y))).unsqueeze(
            1
        )
        y3 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_3(y))).unsqueeze(
            1
        )

        #!HARDCODED Oct 13:  since now default inited with moe, use the first expert as the vanilla vpt, the dim 0 is the dim of experts
        deep_prompt_embds = [
            self.deep_prompt_embeddings_0[0].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_1[0].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_2[0].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_3[0].expand(B, -1, -1, -1),
        ]
        instruction_embds = [y0, y1, y2, y3]

        internal_features = []
        for i in range(len(self.layers)):
            deep_prompt_embd = deep_prompt_embds[i]
            deep_prompt_embd = self.prompt_dropout(deep_prompt_embd)
            x = self.layers[i](x, deep_prompt_embd, instruction_embds[i])
            internal_features.append(x[:, 1:, :])
        f = self.norm(x)  # B L C
        cls_ = self.avgpool(f.transpose(1, 2))  # B C 1
        cls_ = torch.flatten(cls_, 1)
        if return_internal:
            return cls_, f, internal_features
        return cls_, f

    def forward_features_instruct_moe(self, x, y, route_score, return_internal=False):
        """
        x: image
        y: text features
        route_score: [B, n_expert] determine which expert(s) to use. if more than one, interpolate between the prompts
                    [B, n_layer, n_expert] for per-layer routed moe
        return_internal: if true, return the internal features of each layer
        """
        if len(route_score.shape) == 2:
            route_per_layer = False
        elif len(route_score.shape) == 3:
            route_per_layer = True
            # assert route_score.shape[1] == len(self.layers)
        B = x.shape[0]
        x = self.incorporate_prompt(x, cross_feature=y)

        # if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:

        y0 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_0(y))).unsqueeze(
            1
        )
        y1 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_1(y))).unsqueeze(
            1
        )
        y2 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_2(y))).unsqueeze(
            1
        )
        y3 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_3(y))).unsqueeze(
            1
        )
        instruction_embds = [y0, y1, y2, y3]

        all_prompt_experts = [
            self.deep_prompt_embeddings_0[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_1[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_2[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_3[1:, ...].expand(B, -1, -1, -1, -1),
        ]
        static_prompt = [
            self.deep_prompt_embeddings_0[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_1[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_2[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_3[0, ...].expand(B, -1, -1, -1),
        ]
        # construct prompts
        moe_prompt_embds = []

        othro_loss = []
        for i in range(len(self.layers)):
            if route_per_layer:
                # prompt_embeds = torch.einsum(
                #     "bk, bklnh -> blnh", route_score[:, i, :], all_prompt_experts[i]
                # )  # dim 1 is dim of layers
                #!HARDCODED Nov 02: per block
                crt_block_depth = self.depths[i]
                crt_start = sum(self.depths[:i]) - 1
                if i == 0:
                    crt_block_depth -= 1
                    crt_start = 0

                dynamic_prompt = torch.einsum(
                    "blk, bklnh -> blnh",
                    route_score[:, crt_start : crt_start + crt_block_depth, :],
                    all_prompt_experts[i],
                )  # dim 1 is dim of layers
            else:
                # b: batch, k : num_experts, n: number of prompts l: number of blocks in this layer, h: prompt dim
                dynamic_prompt = torch.einsum(
                    "bk, bklnh -> blnh", route_score, all_prompt_experts[i]
                )
                # scale the prompt embeddings (linear interpolation)

            #!HARDCODED Sep 27: concat dynamic and static prompts
            if USE_STATIC == 2:  # 2 means use, 1 is not use
                prompt_embeds = torch.cat(
                    [dynamic_prompt, static_prompt[i]], dim=2
                )  # dim 2 is the number of prompt
            else:
                prompt_embeds = dynamic_prompt


            moe_prompt_embds.append(prompt_embeds)

        internal_features = []
        for i in range(len(self.layers)):
            deep_prompt_embd = moe_prompt_embds[i]
            deep_prompt_embd = self.prompt_dropout(deep_prompt_embd)
            x = self.layers[i](x, deep_prompt_embd, instruction_embds[i])
            # if return_internal:
            internal_features.append(x[:, self.num_prompts :, :])

        f = self.norm(x)  # B L C
        cls_ = self.avgpool(f.transpose(1, 2))  # B C 1
        cls_ = torch.flatten(cls_, 1)
        if return_internal:
            return cls_, internal_features
        # return cls_, f, torch.mean(torch.stack(othro_loss))
        return cls_, f, None

    def forward_features_attn_moe(self, x, y, route_score):
        """
        x: image
        y: text features
        route_score: [B, n_expert] determine which expert(s) to use. if more than one, interpolate between the prompts
        fuse attn values instead of learnable prompts
        """
        B = x.shape[0]
        x = self.incorporate_prompt(x, cross_feature=y)

        # if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:

        y0 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_0(y))).unsqueeze(
            1
        )
        y1 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_1(y))).unsqueeze(
            1
        )
        y2 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_2(y))).unsqueeze(
            1
        )
        y3 = self.prompt_dropout(self.prompt_proj_act(self.prompt_proj_3(y))).unsqueeze(
            1
        )
        instruction_embds = [y0, y1, y2, y3]

        all_prompt_experts = [
            self.deep_prompt_embeddings_0[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_1[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_2[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_3[1:, ...].expand(B, -1, -1, -1, -1),
        ]

        static_prompt = [
            self.deep_prompt_embeddings_0[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_1[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_2[0, ...].expand(B, -1, -1, -1),
            self.deep_prompt_embeddings_3[0, ...].expand(B, -1, -1, -1),
        ]

        # expand and concat all static prompts to prompt experts
        for i in range(len(all_prompt_experts)):
            # expand number_of_expert time and concat
            static_prompt_expanded = (
                static_prompt[i].unsqueeze(1).expand(-1, self.moe_n_experts, -1, -1, -1)
            )
            all_prompt_experts[i] = torch.cat(
                [all_prompt_experts[i], static_prompt_expanded], dim=3
            )  # dim 3 is the number of prompts

        for i in range(len(self.layers)):
            x = self.layers[i](
                x,
                all_prompt_experts[i],
                instruction_embds[i],
                use_attn_fuse=True,
                moe_scores=route_score,
            )

        f = self.norm(x)  # B L C
        cls_ = self.avgpool(f.transpose(1, 2))  # B C 1
        cls_ = torch.flatten(cls_, 1)
        return cls_, f

    def forward_features_instruct_multimodal_moe(self, x, y):
        """
        x: image
        y: text features
        """
        B = x.shape[0]
        self.current_text_feat = y  # store for later use
        self.original_image = x  # 修改处：保存原始图像供每层使用
        # 获取patch embeddings（不包含prompt）
        patch_tokens = self.patch_embed(x)
        if self.ape:
            patch_tokens = patch_tokens + self.absolute_pos_embed
        patch_tokens = self.pos_drop(patch_tokens)
    
        # 第一次调用incorporate_prompt生成初始prompt
        x_with_prompt = self.incorporate_prompt(x, cross_feature=y)
    
        # 计算实际的prompt长度
        total_length = x_with_prompt.shape[1]
        patch_length = patch_tokens.shape[1]
        actual_prompt_length = total_length - patch_length

        # if self.prompt_config.LOCATION == "prepend" and self.prompt_config.DEEP:
        y_ins = self.mm_prompt_instruction_proj(y)
        y0 = self.prompt_dropout(
            self.prompt_proj_act(self.prompt_proj_0(y_ins))
        ).unsqueeze(1)
        y1 = self.prompt_dropout(
            self.prompt_proj_act(self.prompt_proj_1(y_ins))
        ).unsqueeze(1)
        y2 = self.prompt_dropout(
            self.prompt_proj_act(self.prompt_proj_2(y_ins))
        ).unsqueeze(1)
        y3 = self.prompt_dropout(
            self.prompt_proj_act(self.prompt_proj_3(y_ins))
        ).unsqueeze(1)
        instruction_embds = [y0, y1, y2, y3]

        all_prompt_experts = [
            self.deep_prompt_embeddings_0[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_1[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_2[1:, ...].expand(B, -1, -1, -1, -1),
            self.deep_prompt_embeddings_3[1:, ...].expand(B, -1, -1, -1, -1),
        ]

        static_prompt = [
            self.deep_prompt_embeddings_0[0, ...],
            self.deep_prompt_embeddings_1[0, ...],
            self.deep_prompt_embeddings_2[0, ...],
            self.deep_prompt_embeddings_3[0, ...],
        ]

        extra_out = {}
        all_layer_imp_loss = []
        # 修改处：x现在只包含patch tokens
        current_patches = self.patch_embed(x)
        if self.ape:
            current_patches = current_patches + self.absolute_pos_embed
        current_patches = self.pos_drop(current_patches)
        # 修改处：计算特征损失
        if hasattr(self, 'current_mlp_outputs') and self.training:
            crossing_loss = self.feature_loss_computer.feature_crossing_loss(self.current_mlp_outputs['features'])
            self_loss = self.feature_loss_computer.feature_self_loss(
                self.current_mlp_outputs, 
                self.current_multi_level_mlp
            )
            extra_out['crossing_loss'] = crossing_loss
            extra_out['self_loss'] = self_loss

        # 逐层处理
        current_patches = patch_tokens
        # here, default to route per layer
        

        # 逐层处理：每层前重新生成MoPE prompts
        for i in range(len(self.layers)):
            # print(f"\n=== Layer {i} ===")
            # print(f"Input current_patches shape: {current_patches.shape}")
        
            # 1. 每层前重新生成MoPE prompts（19个tokens）
            mope_prompts = self.incorporate_prompt(self.original_image, y)
            # print(f"Generated MoPE prompts shape: {mope_prompts.shape}")
        
            # 2. 投影prompt到当前层维度
            current_dim = int(self.embed_dim * 2 ** i)
            if mope_prompts.shape[-1] != current_dim:
                if not hasattr(self, f'prompt_proj_layer_{i}'):
                    proj_layer = nn.Linear(mope_prompts.shape[-1], current_dim).to(mope_prompts.device)
                    setattr(self, f'prompt_proj_layer_{i}', proj_layer)
                else:
                    proj_layer = getattr(self, f'prompt_proj_layer_{i}')
                mope_prompts = proj_layer(mope_prompts)
                # print(f"Projected MoPE prompts to dim {current_dim}: {mope_prompts.shape}")

            # 3. 投影patch tokens到当前层维度
            if current_patches.shape[-1] != current_dim:
                if not hasattr(self, f'patch_proj_layer_{i}'):
                    patch_proj_layer = nn.Linear(current_patches.shape[-1], current_dim).to(current_patches.device)
                    setattr(self, f'patch_proj_layer_{i}', patch_proj_layer)
                else:
                    patch_proj_layer = getattr(self, f'patch_proj_layer_{i}')
                current_patches = patch_proj_layer(current_patches)
                # print(f"Projected current_patches to dim {current_dim}: {current_patches.shape}")

            # 4. 拼接MoPE prompts和patch tokens
            x_with_prompts = torch.cat([mope_prompts, current_patches], dim=1)
            # print(f"Concatenated x_with_prompts shape: {x_with_prompts.shape}")
        
            # 5. 调用当前layer（返回可能是tuple）
            if i < len(all_prompt_experts) and i < len(static_prompt):
                layer_output = self.layers[i](
                    x_with_prompts,
                    deep_prompt_embd=None,
                    instruct_prompt_embd=instruction_embds[i],
                    use_mm_fuse=True,
                    cross_feature=y,
                    all_prompt_experts=all_prompt_experts[i],
                    static_prompt=static_prompt[i],
                )
            else:
                layer_output = self.layers[i](x_with_prompts)
        
            # 6. 处理layer输出（可能是tuple）
            if isinstance(layer_output, tuple):
                current_output, imp_losses = layer_output
                if isinstance(imp_losses, list):
                    all_layer_imp_loss.extend(imp_losses)
                else:
                    all_layer_imp_loss.append(imp_losses)
            else:
                current_output = layer_output
            
            # print(f"Layer {i} output shape: {current_output.shape}")
        
            # 7. **关键**：移除prompt部分，只保留patch tokens传给下一层
            prompt_length = mope_prompts.shape[1]
            current_patches = current_output[:, prompt_length:, :]  # 只保留patch部分
            # print(f"Extracted current_patches for next layer: {current_patches.shape}")

        # 最后一层的输出包含prompt（用于最终classification）
        final_output = current_output  # 保留完整输出用于classification
        
        if all_layer_imp_loss:
            extra_out["importance_loss"] = torch.mean(torch.stack(all_layer_imp_loss))
        f = self.norm(final_output)  # B L C
        cls_ = self.avgpool(f.transpose(1, 2))  # B C 1
        cls_ = torch.flatten(cls_, 1)
        return cls_, f, extra_out

    def forward(self, x):
        cls_, f = self.forward_features(x)
        return f

    def load_state_dict(self, state_dict, strict):

        super(PromptedSwinTransformer, self).load_state_dict(state_dict, strict)



class PromptedPatchMerging(PatchMerging):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        num_prompts,
        prompt_location,
        deep_prompt,
        input_resolution,
        dim,
        norm_layer=nn.LayerNorm,
    ):
        super(PromptedPatchMerging, self).__init__(input_resolution, dim, norm_layer)
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        if prompt_location == "prepend":
            if not deep_prompt:
                self.prompt_upsampling = None
                # self.prompt_upsampling = nn.Linear(dim, 4 * dim, bias=False)
            else:
                self.prompt_upsampling = None

    def upsample_prompt(self, prompt_emb):
        if self.prompt_upsampling is not None:
            prompt_emb = self.prompt_upsampling(prompt_emb)
        else:
            prompt_emb = torch.cat(
                (prompt_emb, prompt_emb, prompt_emb, prompt_emb), dim=-1
            )
        return prompt_emb

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        if self.prompt_location == "prepend":
            # 修改处：动态计算实际的prompt数量
            expected_patch_tokens = H * W
            actual_num_prompts = L - expected_patch_tokens
            # print(f"DEBUG - PromptedPatchMerging: input_shape={x.shape}, expected_patches={expected_patch_tokens}, actual_prompts={actual_num_prompts}, self.num_prompts={self.num_prompts}")
        
            # change input size
            prompt_emb = x[:, :actual_num_prompts, :]
            x = x[:, actual_num_prompts:, :]
            L = L - actual_num_prompts
            prompt_emb = self.upsample_prompt(prompt_emb)
            assert L == H * W, "input feature has wrong size, should be {}, got {}".format(
                H * W, L
            )
            assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # add the prompt back:
        if self.prompt_location == "prepend":
            x = torch.cat((prompt_emb, x), dim=1)

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PromptedSwinTransformerBlock(SwinTransformerBlock):
    def __init__(
        self,
        num_prompts,
        prompt_location,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        d_cross=0,
        d_inter=0,
        moe_n_experts=0,
        is_last=False,
    ):
        super(PromptedSwinTransformerBlock, self).__init__(
            dim,
            input_resolution,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            act_layer,
            norm_layer,
        )
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location
        # 修改处：添加调试信息
        # print(f"DEBUG - PromptedSwinTransformerBlock __init__:")
        # print(f"  num_prompts passed: {num_prompts}")
        # print(f"  prompt_location: {prompt_location}")
        self.is_last = is_last
        if self.prompt_location == "prepend":
            self.attn = PromptedWindowAttention(
                num_prompts,
                prompt_location,
                dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        
        self.mm_prompt_global_pooler = nn.AdaptiveAvgPool1d(1)
        self.mm_prompt_self_proj = nn.Linear(dim, d_inter)
        self.mm_prompt_cross_proj = nn.Linear(768, d_cross)
        self.mm_frozen_expert_key_embed = nn.Parameter(
            torch.zeros(d_cross + d_inter, moe_n_experts)
        )
        # orthogonal initialize key_embed
        nn.init.orthogonal_(self.mm_frozen_expert_key_embed)
        self.mm_frozen_expert_key_embed.requires_grad = False

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        # if self.prompt_location == "prepend":
        #     # change input size
        #     prompt_emb = x[:, :num_prompts, :]
        #     x = x[:, num_prompts:, :]
        #     L = L - num_prompts
        # 修改处：确保L是正确的patch数量
        # 修改处：动态计算prompt数量，但保持处理逻辑简单
        expected_patch_tokens = H * W
        if L > expected_patch_tokens:
            actual_num_prompts = L - expected_patch_tokens
            prompt_emb = x[:, :actual_num_prompts, :]
            x_patches = x[:, actual_num_prompts:, :]
        else:
            actual_num_prompts = 0
            prompt_emb = None
            x_patches = x
            
        # print(f"DEBUG - PromptedSwinTransformerBlock.forward:")
        # print(f"  Input shape: {x.shape}")
        # print(f"  H={H}, W={W}, expected_patch_tokens={expected_patch_tokens}")
        # print(f"  actual_num_prompts: {actual_num_prompts}")
        # print(f"  self.num_prompts (from init): {self.num_prompts}")
        # if prompt_emb is not None:
        #     print(f"  prompt_emb shape: {prompt_emb.shape}")
        # print(f"  x_patches shape: {x_patches.shape}")
        # assert L == H * W, "input feature has wrong size, should be {}, got {}".format(
        #     H * W, L
        # )
        assert x_patches.shape[1] == H * W, f"patch tokens should be {H * W}, got {x_patches.shape[1]}"
        shortcut = x_patches  # 只对patch部分做残差连接
        x_patches = self.norm1(x_patches)
        # print("x_patches shape:", x_patches.shape)
        x_patches = x_patches.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x_patches, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x_patches

        # partition windows --> nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C

        # add back the prompt for attn for parralel-based prompts
        # nW*B, num_prompts + window_size*window_size, C
        num_windows = int(x_windows.shape[0] / B)
        if self.prompt_location == "prepend" and prompt_emb is not None and actual_num_prompts > 0:
            # 将prompt复制到每个窗口
            prompt_expanded = prompt_emb.unsqueeze(1).expand(B, num_windows, -1, -1)  # [B, nW, num_prompts, C]
            prompt_expanded = prompt_expanded.contiguous().view(-1, actual_num_prompts, C)  # [nW*B, num_prompts, C]
            x_windows = torch.cat([prompt_expanded, x_windows], dim=1)
            # print(f"  prompt_expanded shape: {prompt_expanded.shape}")
            # print(f"  x_windows shape after prompt: {x_windows.shape}")

        
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        # print(f"  attn_windows shape: {attn_windows.shape}")
        # attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # # seperate prompt embs --> nW*B, num_prompts, C
        # if self.prompt_location == "prepend" and prompt_emb is not None:
        #     # change input size
        #     prompt_emb = attn_windows[:, :num_prompts, :]
        #     attn_windows = attn_windows[:, num_prompts:, :]
        #     # change prompt_embs's shape:
        #     # nW*B, num_prompts, C - B, num_prompts, C
        #     prompt_emb = prompt_emb.view(-1, B, num_prompts, C)
        #     prompt_emb = prompt_emb.mean(0)
        
        # 修改处：修复变量名不一致的问题
        if self.prompt_location == "prepend" and prompt_emb is not None and actual_num_prompts > 0:
            prompt_emb_out = attn_windows[:, :actual_num_prompts, :]  # [nW*B, actual_num_prompts, C]
            attn_windows_patches = attn_windows[:, actual_num_prompts:, :]  # [nW*B, window_size^2, C]
            
            # print(f"  prompt_emb_out shape before view: {prompt_emb_out.shape}")
            # print(f"  Trying to reshape: [nW*B={prompt_emb_out.shape[0]}, actual_num_prompts={actual_num_prompts}, C={C}]")
            # print(f"  Expected: [{num_windows}, {B}, {actual_num_prompts}, {C}]")
            
            # 修改处：使用actual_num_prompts而不是self.num_prompts或num_prompts
            prompt_emb_out = prompt_emb_out.view(num_windows, B, actual_num_prompts, C).mean(0)
            # print(f"  prompt_emb_out shape after view and mean: {prompt_emb_out.shape}")
        else:
            prompt_emb_out = None
            attn_windows_patches = attn_windows

        # merge windows
        attn_windows_patches = attn_windows_patches.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows_patches, self.window_size, H, W)  # B H W C

        # reverse cyclic shift
        if self.shift_size > 0:
            x_patches = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x_patches = shifted_x
        x_patches = x_patches.view(B, H * W, C)

        # FFN
        x_patches = shortcut + self.drop_path(x_patches)
        x_patches = x_patches + self.drop_path(self.mlp(self.norm2(x_patches)))
        # add the prompt back:
        # 修改处：重新拼接prompt和patch tokens
        if prompt_emb_out is not None:
            x = torch.cat((prompt_emb_out, x_patches), dim=1)
            # print(f"  Final result shape: {x.shape}")
        else:
            x = x_patches
            # print(f"  Final result shape (no prompts): {x.shape}")

        return x

    def forward_attn_moe(
        self, x_input, all_prompt_experts, instruct_prompt_embd, moe_scores
    ):
     raise NotImplementedError

    def forward_mm_moe(
        self,
        x_input,
        static_prompt,
        all_prompt_experts,
        instruct_prompt_embd,
        cross_feature,
    ):
        """
        The main forward function for multimodal moe (MoPE)
        """
        
        #修改处
        # *** Modified: Align batch dimensions across all prompt components ***
        # 修改处: 确保 x_input 为 3D 张量 (B, L, C)，展平额外维度（若存在）
        if x_input.dim() != 3:
            x_input = x_input.view(x_input.size(0), -1, x_input.size(-1))
        B_img = x_input.size(0)
        B_static = static_prompt.size(0) if static_prompt is not None else B_img
        B_expert = all_prompt_experts.size(0) if all_prompt_experts is not None else B_img
        B_inst = instruct_prompt_embd.size(0) if instruct_prompt_embd is not None else B_img
        B_cross = cross_feature.size(0) if cross_feature is not None else B_img
        max_B = max(B_img, B_static, B_expert, B_inst, B_cross)
        # If any component has fewer samples, pad it with zeros to match max_B
        if not (B_img == B_static == B_expert == B_inst == B_cross):
            if B_img < max_B:
                pad_count = max_B - B_img
                pad_x = torch.zeros(pad_count, x_input.size(1), x_input.size(2), device=x_input.device)
                x_input = torch.cat([x_input, pad_x], dim=0)
            if static_prompt is not None and B_static < max_B:
                pad_count = max_B - B_static
                pad_static = torch.zeros(pad_count, static_prompt.size(1), static_prompt.size(2), device=static_prompt.device)
                static_prompt = torch.cat([static_prompt, pad_static], dim=0)
            if all_prompt_experts is not None and B_expert < max_B:
                pad_count = max_B - B_expert
                pad_experts = torch.zeros(
                    pad_count, 
                    all_prompt_experts.size(1), all_prompt_experts.size(2), all_prompt_experts.size(3),
                    device=all_prompt_experts.device
                )
                all_prompt_experts = torch.cat([all_prompt_experts, pad_experts], dim=0)
            if instruct_prompt_embd is not None and B_inst < max_B:
                pad_count = max_B - B_inst
                pad_instruct = torch.zeros(pad_count, instruct_prompt_embd.size(1), instruct_prompt_embd.size(2), device=instruct_prompt_embd.device)
                instruct_prompt_embd = torch.cat([instruct_prompt_embd, pad_instruct], dim=0)
            if cross_feature is not None and B_cross < max_B:
                pad_count = max_B - B_cross
                pad_cross = torch.zeros(pad_count, cross_feature.size(1), device=cross_feature.device)
                cross_feature = torch.cat([cross_feature, pad_cross], dim=0)
            # After padding, all components have batch size = max_B (safe for concatenation)
        #修改处

        # 修改处: 确保 cross_feature 为 2D 张量 (B, embed_dim)，若有多维则取第一个元素（CLS）
        if cross_feature is not None and cross_feature.dim() > 2:
            cross_feature = cross_feature[:, 0, :]
            
        # first synthesis the prompt embd
        x_pooled = self.mm_prompt_global_pooler(x_input.transpose(1, 2)).squeeze(
            -1
        )  # B, C

        moe_self_embd = self.mm_prompt_self_proj(x_pooled)  # B, D_moe_embd
         # 修改处：确保cross_feature维度正确 - 应该是768维的原始文本特征
        if cross_feature is not None:
            # 如果cross_feature已经被投影过，我们需要使用原始的768维特征
            # 这里假设传入的cross_feature是原始的768维BERT特征
            moe_cross_embd = self.mm_prompt_cross_proj(cross_feature)  # B, D_moe_embd
        else:
            # 无文本时，创建零向量
            moe_cross_embd = torch.zeros_like(moe_self_embd)
        moe_joint_embd = torch.cat(
            [moe_self_embd, moe_cross_embd], dim=1
        )  # B,  D_moe_embd *2
        # moe_joint_embd = moe_cross_embd
        # B,  D_moe_embd *2
        # get the logit by dot product with expert key at this layer  B,  D_moe_embd *2 @ D_moe_embd *2 , k_expert -> B, k_expert
        moe_logits = moe_joint_embd @ self.mm_frozen_expert_key_embed  # B, k_expert # B, k_expert
        temperature = 0.1
        # add normal dis N(0, 1/n_experts^2)
        # add gumbel noise
        
        noise = torch.randn(moe_logits.shape).to(moe_logits.device) / (16**2)
        moe_logits = moe_logits / temperature

        if "multi-label mode":
            moe_scores = torch.sigmoid(moe_logits)  # independent activation of experts
        else:
            moe_scores = F.softmax(moe_logits + noise, dim=-1)
        moe_mask = torch.zeros(moe_scores.shape).to(moe_scores.device)
        # if self.route_per_layer:
        # get the dynamic prompt
        if all_prompt_experts is not None:
            dynamic_prompt = torch.einsum("bk, bknh -> bnh", moe_scores, all_prompt_experts)
        else:
            # 如果没有专家prompt，创建零向量
            dynamic_prompt = torch.zeros(max_B, 1, x_input.size(-1), device=x_input.device)
            
        # get the prompt embedding for this forward, which is concat of all
        prompt_parts = []
        if static_prompt is not None:
            prompt_parts.append(static_prompt)
        if dynamic_prompt is not None:
            prompt_parts.append(dynamic_prompt)
        if instruct_prompt_embd is not None:
            prompt_parts.append(instruct_prompt_embd)
            
        if prompt_parts:
            # 确保所有prompt_parts都是3维张量
            fixed_prompt_parts = []
            for i, part in enumerate(prompt_parts):
                if part.dim() == 4:
                    # 如果是4维，重塑为3维 [B, L, C]
                    part = part.view(part.size(0), -1, part.size(-1))
                elif part.dim() == 2:
                    # 如果是2维，增加一个维度
                    part = part.unsqueeze(1)
                elif part.dim() == 3:
                    # 已经是3维，保持不变
                    pass
                else:
                    raise ValueError(f"Unexpected tensor dimension: {part.dim()}")
            fixed_prompt_parts.append(part)

            prompt_emb = torch.cat(fixed_prompt_parts, dim=1)
        else:
            # 如果没有任何prompt，创建零向量
            prompt_emb = torch.zeros(max_B, 1, x_input.size(-1), device=x_input.device)
        # # get the prompt embedding for this forward, which is concat of all
        # prompt_emb = torch.cat(
        #     (static_prompt, dynamic_prompt, instruct_prompt_embd), dim=1
        # )

        sum_scores = torch.sum(moe_scores, dim=0)  # N_expert
        std_scores = torch.std(sum_scores, dim=-1)  # 1
        mean_scores = torch.mean(sum_scores, dim=-1)  # 1
        threshold = 0.1
        importance_loss = (std_scores / mean_scores) ** 2 if mean_scores > 0 else 0
        # importance_loss = torch.where(importance_loss > threshold, importance_loss, torch.zeros_like(importance_loss))
        #importance_loss = (
        #    importance_loss
        #    if importance_loss > threshold
        #    else 0 #torch.zeros_like(importance_loss)
        #)
        #修改处
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        importance_loss = torch.tensor(importance_loss).to(device)
        if importance_loss > threshold:
            importance_loss = importance_loss
        else:
            importance_loss = torch.tensor(0.0).to(device)
        # imp_loss = torch.mean(importance_loss)

        # the following is same as forward.
        # 修改处：确保拼接时维度匹配
        if prompt_emb.size(1) + x_input.size(1) <= x_input.size(1):
            # 避免重复拼接
            x = torch.cat((prompt_emb, x_input), dim=1)
        else:
            # 如果x_input已经包含了prompt，只取后面的部分
            x = torch.cat((prompt_emb, x_input[:, prompt_emb.size(1):, :]), dim=1)
        return self.forward(x), importance_loss


class PromptedWindowAttention(WindowAttention):
    def __init__(
        self,
        num_prompts,
        prompt_location,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super(PromptedWindowAttention, self).__init__(
            dim, window_size, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop
        )
        self.num_prompts = num_prompts
        self.prompt_location = prompt_location

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        

        B_, N, C = x.shape
        # 修改处：动态计算实际的prompt数量
        expected_patch_tokens = self.window_size[0] * self.window_size[1]
        actual_num_prompts = N - expected_patch_tokens if N > expected_patch_tokens else 0
        # print(f"DEBUG - PromptedWindowAttention:")
        # print(f"  Input shape: {x.shape}")
        # print(f"  Window size: {self.window_size}")
        # print(f"  Expected patch tokens: {expected_patch_tokens}")
        # print(f"  Actual num prompts: {actual_num_prompts}")
        # print(f"  self.num_prompts (from init): {self.num_prompts}")

        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # print(f"  attn shape before bias: {attn.shape}")

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww

        # account for prompt nums for relative_position_bias
        # attn: [1920, 6, 649, 649]
        # relative_position_bias: [6, 49, 49])

        if self.prompt_location == "prepend" and actual_num_prompts > 0:
            # expand relative_position_bias
            _C, _H, _W = relative_position_bias.shape
            # print(f"  Before expansion: _C={_C}, _H={_H}, _W={_W}")

            # 第一次扩展：添加prompt行
            prompt_rows = torch.zeros(_C, actual_num_prompts, _W, device=attn.device)
            relative_position_bias = torch.cat([prompt_rows, relative_position_bias], dim=1)
            # print(f"  After adding prompt rows: {relative_position_bias.shape}")
            
            # 第二次扩展：添加prompt列
            _C, _H_new, _W = relative_position_bias.shape
            prompt_cols = torch.zeros(_C, _H_new, actual_num_prompts, device=attn.device)
            relative_position_bias = torch.cat([prompt_cols, relative_position_bias], dim=-1)
            # print(f"  After adding prompt cols: {relative_position_bias.shape}")

        # print(f"  Final relative_position_bias shape: {relative_position_bias.shape}")
        # print(f"  Final attn shape: {attn.shape}")
        
        # 修改处：检查维度匹配
        if relative_position_bias.shape[-1] != attn.shape[-1]:
            # print(f"  ERROR: Dimension mismatch!")
            # print(f"    attn last dim: {attn.shape[-1]}")  
            # print(f"    bias last dim: {relative_position_bias.shape[-1]}")
            
            # 修改处：强制修正relative_position_bias的大小
            expected_size = attn.shape[-1]
            if relative_position_bias.shape[-1] > expected_size:
                # 如果bias太大，截取
                relative_position_bias = relative_position_bias[:, :expected_size, :expected_size]
                # print(f"    Truncated bias to: {relative_position_bias.shape}")
            elif relative_position_bias.shape[-1] < expected_size:
                # 如果bias太小，填充
                _C, _H, _W = relative_position_bias.shape
                pad_size = expected_size - _W
                pad_bias = torch.zeros(_C, _H, pad_size, device=attn.device)
                relative_position_bias = torch.cat([relative_position_bias, pad_bias], dim=-1)
                
                _C, _H, _W = relative_position_bias.shape
                pad_size = expected_size - _H
                pad_bias = torch.zeros(_C, pad_size, _W, device=attn.device)
                relative_position_bias = torch.cat([relative_position_bias, pad_bias], dim=1)
                # print(f"    Padded bias to: {relative_position_bias.shape}")

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # incorporate prompt
            # mask: (nW, 49, 49) --> (nW, 49 + n_prompts, 49 + n_prompts)
            nW = mask.shape[0]
            if self.prompt_location == "prepend" and actual_num_prompts > 0:
                # 修改处：根据实际prompt数量调整mask
                if mask.shape[1] != N or mask.shape[2] != N:
                    new_mask = torch.zeros(nW, N, N, device=mask.device)
                    # prompt部分允许全连接
                    new_mask[:, :actual_num_prompts, :] = 0
                    new_mask[:, :, :actual_num_prompts] = 0
                    # patch部分使用原始mask
                    patch_size = min(mask.shape[1], N - actual_num_prompts)
                    if patch_size > 0:
                        new_mask[:, actual_num_prompts:actual_num_prompts+patch_size, 
                                actual_num_prompts:actual_num_prompts+patch_size] = mask[:, :patch_size, :patch_size]
                    mask = new_mask
            # logger.info("before", attn.shape)
            # attn: B batch size for input image, nW: number of windows, nH: number of heads, N: window size (with prompt), N: window size (with prompt)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            # logger.info("after", attn.shape)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_moe(self, xs, mask=None, moe_scores=None):
        """
        Args:
            xs: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        all_x = []  # all x after self-attn
        for x in xs:
            B_, N, C = x.shape
            qkv = (
                self.qkv(x)
                .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )  # make torchscript happy (cannot use tensor as tuple)

            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1
            ).contiguous()  # nH, Wh*Ww, Wh*Ww

            # account for prompt nums for relative_position_bias
            # attn: [1920, 6, 649, 649]
            # relative_position_bias: [6, 49, 49])

            if self.prompt_location == "prepend":
                # expand relative_position_bias
                _C, _H, _W = relative_position_bias.shape

                relative_position_bias = torch.cat(
                    (
                        torch.zeros(_C, self.num_prompts, _W, device=attn.device),
                        relative_position_bias,
                    ),
                    dim=1,
                )
                relative_position_bias = torch.cat(
                    (
                        torch.zeros(
                            _C,
                            _H + self.num_prompts,
                            self.num_prompts,
                            device=attn.device,
                        ),
                        relative_position_bias,
                    ),
                    dim=-1,
                )

            attn = attn + relative_position_bias.unsqueeze(0)
            # no mask here
            attn = self.softmax(attn)

            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            all_x.append(x)

        # blend by moe scores
        x = torch.stack(all_x, dim=1)  # B, n_experts, H*W, C
        # b: batch size, k: k experts in total, l: sequence length, h: hidden dim
        x = torch.einsum("bk, bklh -> blh", moe_scores, x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_swin_encoder(
    num_classes,
    crop_size,
    use_vpt=True,
    moe_n_experts=8,
    prompt_length=10,
    use_static_prompt=False,
    use_instruct=True,
    prompt_init="uniform",
    d_cross=0,
    d_inter=0,
):
    if use_vpt:
        if crop_size == 224:
            model = PromptedSwinTransformer(
                img_size=crop_size,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=7,
                drop_path_rate=0.5,
                num_classes=num_classes,
                moe_n_experts=moe_n_experts,
                prompt_length=prompt_length,
                use_static_prompt=use_static_prompt,
                use_instruct=use_instruct,
                prompt_init=prompt_init,
                d_cross=d_cross,
                d_inter=d_inter,
            )
        elif crop_size == 384:
            model = PromptedSwinTransformer(
                img_size=crop_size,
                embed_dim=128,
                depths=[2, 2, 18, 2],
                num_heads=[4, 8, 16, 32],
                window_size=12,
                drop_path_rate=0.2,
                num_classes=num_classes,
                moe_n_experts=moe_n_experts,
                prompt_length=prompt_length,
                use_static_prompt=use_static_prompt,
                use_instruct=use_instruct,
            )
        elif crop_size == 512:
            model = PromptedSwinTransformer(
                img_size=crop_size,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                drop_path_rate=0.3,
                num_classes=num_classes,
                moe_n_experts=moe_n_experts,
                prompt_length=prompt_length,
                use_static_prompt=use_static_prompt,
                use_instruct=use_instruct,
            )

        # freeze all parameter except name with "prompt" or "moe"
        for name, param in model.named_parameters():
            if "prompt" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    else:
        model = SwinTransformer(
            img_size=crop_size,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.5,
            num_classes=num_classes,
        )

    # load checkpoint
    if crop_size == 224:
        model_w = "pretrained/swin_base_patch4_window7_224_22k.pth"
    elif crop_size == 384:
        model_w = "pretrained/swin_base_patch4_window12_384_22k.pth"
    elif crop_size == 512:
        model_w = "/pretrained/upernet_swin_small_patch4_window7_512x512.pth"
    checkpoint = torch.load(model_w, map_location="cpu")
    state_dict = checkpoint["model"]
    # ignore head weight when loading
    state_dict.pop("head.weight")
    state_dict.pop("head.bias")
    model.load_state_dict(state_dict, strict=False)
    return model


def get_swin_classifier(
    num_classes,
    backbone,
    use_vpt,
    moe_n_experts,
    prompt_length=10,
    use_static_prompt=False,
    prompt_init="uniform",
    use_instruct=True,
    d_cross=0,
    d_inter=0,
):
    class SwinClassifier(nn.Module):
        def __init__(
            self,
            num_classes,
            use_vpt,
            moe_n_experts=8,
            prompt_length=10,
            use_static_prompt=False,
            prompt_init="uniform",
            use_instruct=True,
            d_cross=0,
            d_inter=0,
        ):
            super(SwinClassifier, self).__init__()
            crop_size = 224
            if "384" in backbone:
                crop_size = 384
            self.encoder = get_swin_encoder(
                num_classes,
                crop_size,
                use_vpt,
                moe_n_experts,
                prompt_length,
                use_static_prompt,
                prompt_init=prompt_init,
                use_instruct=use_instruct,
                d_cross=d_cross,
                d_inter=d_inter,
            )
            self.classifier = nn.Linear(1024, num_classes)

        def forward(self, x, return_features=False):
            cls_, _ = self.encoder.forward_features(x)
            x = self.classifier(cls_)
            if not return_features:
                return x
            else:
                return x, cls_

        def forward_instruct(
            self,
            x,
            text_feature=None,
            return_features=False,
        ):
            """
            project forzen text encoder cls as prompt
            """
            cls_, _ = self.encoder.forward_features_instruct(x, text_feature)
            x = self.classifier(cls_)
            if not return_features:
                return x
            else:
                return x, cls_

        def forward_instruct_moe(
            self, x, text_feature, route_score, return_features=False, attn_fuse=False
        ):
            if attn_fuse:
                cls_, _ = self.encoder.forward_features_attn_moe(
                    x, text_feature, route_score
                )
                extra_loss = 0.0
            else:
                cls_, _, extra_loss = self.encoder.forward_features_instruct_moe(
                    x, text_feature, route_score
                )
            x = self.classifier(cls_)
            if not return_features:
                return x, extra_loss
            else:
                return x, cls_, extra_loss

        def forward_instruct_multimodal_moe(
            self, x, text_feature, return_features=False
        ):
            """
            moe where the prompt vector is conditioned both on pooled img feature and the text feature.
            """
            cls_, _, extra_out = self.encoder.forward_features_instruct_multimodal_moe(
                x, text_feature
            )
            x = self.classifier(cls_)
            extra_out["cls"] = cls_
            if not return_features:
                return x, extra_out
            else:
                return x, cls_, extra_out



    return SwinClassifier(
        num_classes,
        use_vpt,
        moe_n_experts,
        prompt_length,
        use_static_prompt,
        prompt_init,
        use_instruct,
        d_cross=d_cross,
        d_inter=d_inter,
    )


def get_swin_segmentor(
    num_classes,
    crop_size,
    use_vpt,
    moe_n_experts,
    prompt_length=10,
    use_static_prompt=False,
):
    class SwinSegmentor(nn.Module):
        def __init__(
            self,
            num_classes,
            use_vpt,
            moe_n_experts=8,
            prompt_length=10,
            use_static_prompt=False,
        ):
            super(SwinSegmentor, self).__init__()
            self.encoder = get_swin_encoder(
                num_classes,
                crop_size,
                use_vpt,
                moe_n_experts,
                prompt_length,
                use_static_prompt,
            )
            # self.segmentor = SeTRPUPHead(crop_size, num_classes=num_classes, prompt_len= self.encoder.num_prompts)
            self.segmentor = SwinUperNetHead()
            # self.instruct_learned_down_sampler = nn.Linear(145, 144)

        def forward(self, x, return_features=False):
            cls_, f = self.encoder.forward_features(x)
            x = self.segmentor(f)
            if not return_features:
                return x
            else:
                return x, f

        def forward_instruct(self, x, text_feature, return_features=False, **kwargs):
            cls_, f, fs = self.encoder.forward_features_instruct(
                x, text_feature, return_internal=True
            )

            x = self.segmentor(fs)
            if not return_features:
                return x
            else:
                return x, f

        # TODO Oct 13:
        def forward_late_concat(self, x, text_feature, return_features=False, **kwargs):
            cls_, f = self.encoder.forward_features(x)
            # repeat text feature and add to f
            text_feature = text_feature.unsqueeze(1)
            text_feature = text_feature.expand(-1, -1, f.shape[1])
            f = f + text_feature

            x = self.segmentor(f)
            if not return_features:
                return x
            else:
                return x, f

        def forward_instruct_moe(
            self, x, text_feature, route_score, return_features=False, attn_fuse=False
        ):
            if attn_fuse:
                cls_, f = self.encoder.forward_features_attn_moe(
                    x, text_feature, route_score
                )
            else:
                cls_, fs = self.encoder.forward_features_instruct_moe(
                    x, text_feature, route_score, return_internal=True
                )
            x = self.segmentor(fs)
            if not return_features:
                return x
            else:
                return x, f

        def slide_inference(self, x, crop_size, stride):
            """
            x: B, C, H, W
            crop_size: [crop_h, crop_w]
            stride: [stride_h, stride_w]
            """
            B, C, H, W = x.shape
            crop_h, crop_w = crop_size
            stride_h, stride_w = stride
            assert (
                H % stride_h == 0 and W % stride_w == 0
            ), "input feature has wrong size, should be {}, got {}".format(H * W, L)
            assert (
                crop_h % stride_h == 0 and crop_w % stride_w == 0
            ), "crop size has wrong size, should be {}, got {}".format(H * W, L)
            assert (
                crop_h <= H and crop_w <= W
            ), "crop size should be smaller than input feature size"

            # crop
            ret = []
            for i in range(0, H - crop_h + 1, stride_h):
                for j in range(0, W - crop_w + 1, stride_w):
                    crop = x[:, :, i : i + crop_h, j : j + crop_w]
                    crop_ret = self.forward(crop)
                    ret.append(crop_ret)
            ret = torch.stack(ret, dim=1)
            ret = ret.view(B, -1, num_classes)
            return ret

    return SwinSegmentor(
        num_classes, use_vpt, moe_n_experts, prompt_length, use_static_prompt
    )


if __name__ == "__main__":
    crop_size = 224
    model = get_swin_encoder(crop_size, use_vpt=True)
    model.eval()

    dummy_input = torch.randn(1, 3, crop_size, crop_size)

    seg = get_swin_segmentor(20)
    ret = seg(dummy_input)
    dummy_input_2 = torch.randn(1, 3, crop_size, 600)
    ret = seg.slide_inference(dummy_input_2, [crop_size, crop_size], [128, 128])

    print(ret.shape)
