import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, i):
        i = self.fc1(i)
        i = self.act(i)
        i = self.drop(i)
        i = self.fc2(i)
        i = self.drop(i)
        return i


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LDBlock(nn.Module):
    """
    Implementation of LD Transformer Block
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, dilation=8, LD=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, cpe_per_block=False):
        super().__init__()
        self.gsm = None
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.local_size = window_size
        self.mlp_ratio = mlp_ratio
        self.LD = LD
        self.dilation = dilation
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, (3, 3), padding=1, groups=dim)
        if min(self.input_resolution) <= self.local_size:
            # if window size is larger than input resolution, we don't partition windows
            self.local_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.local_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        H, W = self.input_resolution

        shortcut = x
        x = self.norm1(x)

        if H < self.local_size:
            self.local_size = (H, W)

        B, Hx, Wx, C = x.shape
        Lc = self.local_size
        Lc_tup = to_2tuple(Lc)

        if self.LD == 0:  # L-MSA

            x = x.view(B, Hx // Lc, Lc, Wx // Lc, Lc, C).permute(0, 1, 3, 2, 4, 5).contiguous()
            x1 = x.reshape(-1, reduce(mul, Lc_tup), C)

        else:  # D-MSA
            dilation = self.dilation

            num_patches_h = Hx // Lc
            num_patches_w = Wx // Lc

            patches = []

            # Extract the patches from the tensor
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    # Calculate the starting indices for the current patch
                    start_h = i
                    start_w = j

                    # Calculate the ending indices for the current patch
                    end_h = start_h + (Lc - 1) * dilation + 1
                    end_w = start_w + (Lc - 1) * dilation + 1

                    # Extract the patch using slicing
                    patch = x[:, start_h:end_h:dilation, start_w:end_w:dilation, :]

                    # Append the patch to the list
                    patches.append(patch)

            # Convert the list of patches to a tensor
            patches_tensor = torch.stack(patches, dim=1)
            x1 = patches_tensor.contiguous().view(-1, reduce(mul, Lc_tup), C)

        attn_mask = None

        # multi-head self-attention
        x = self.attn(x1, mask=attn_mask)

        # ungroup embeddings
        if self.LD == 0:
            x = x.reshape(B, Hx // Lc, Wx // Lc, Lc, Lc,
                          C).permute(0, 1, 3, 2, 4, 5).contiguous()  # B, Hp//G, G, Wp//G, G, C
            x = x.view(B, Hx, Wx, -1)
        else:
            num_patches = num_patches_h * num_patches_w
            x = x.reshape(B, num_patches, Lc, Lc, C).permute(0, 1, 2, 3, 4).contiguous()
            reconstructed_tensor = shortcut.detach().clone()
            for i in range(num_patches):
                # Calculate the row and column indices for the current patch
                row = i // int(num_patches ** 0.5)
                col = i % int(num_patches ** 0.5)

                # Calculate the starting and ending indices for the current patch
                start_h = row
                start_w = col
                end_h = start_h + (Lc * dilation)
                end_w = start_w + (Lc * dilation)

                # Extract the current patch from the patches tensor
                patch = x[:, i, :, :, :]

                # Insert the patch into the appropriate location in the reconstructed tensor
                reconstructed_tensor[:, start_h:end_h:dilation, start_w:end_w:dilation, :] = patch

            x = reconstructed_tensor.view(B, Hx, Wx, -1)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x


class GlobalAttention(nn.Module):
    """Implementation of self-attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalBlock(nn.Module):
    """
    Implementation of Global Transformer Block
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_block=False):
        super().__init__()
        self.cpe_per_block = cpe_per_block
        if self.cpe_per_block:
            self.pos_embed = nn.Conv2d(dim, dim, (3, 3), padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = GlobalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale, attn_drop=attn_drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        if self.cpe_per_block:
            x = x + self.pos_embed(x)
        x = x.permute(0, 2, 3, 1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.permute(0, 3, 1, 2)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.
    """

    def __init__(self, img_size=224, in_chans=3, hidden_dim=16,
                 patch_size=4, embed_dim=96, patch_way=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        self.img_size = img_size
        assert patch_way in ['overlaping', 'nonoverlaping', 'pointconv'], \
            "the patch embedding way isn't exist!"
        if patch_way == "nonoverlaping":
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif patch_way == "overlaping":
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=(3, 3), stride=(1, 1),
                          padding=1, bias=False),  # 224x224
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, int(hidden_dim * 2), kernel_size=(3, 3), stride=(2, 2),
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=(3, 3), stride=(1, 1),
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 4), embed_dim, kernel_size=(3, 3), stride=(2, 2),
                          padding=1, bias=False),  # 56x56
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_chans, hidden_dim, kernel_size=(3, 3), stride=(2, 2),
                          padding=1, bias=False),  # 112x112
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(hidden_dim, int(hidden_dim * 2), kernel_size=(1, 1), stride=(1, 1),
                          padding=0, bias=False),  # 112x112
                nn.BatchNorm2d(int(hidden_dim * 2)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 2), int(hidden_dim * 4), kernel_size=(3, 3), stride=(2, 2),
                          padding=1, bias=False),  # 56x56
                nn.BatchNorm2d(int(hidden_dim * 4)),
                nn.GELU(),
                nn.Conv2d(int(hidden_dim * 4), embed_dim, kernel_size=(1, 1), stride=(1, 1),
                          padding=0, bias=False),  # 56x56
            )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # B, C, H, W
        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(self, in_channels, out_channels, merging_way, cpe_per_satge, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert merging_way in ['conv3_2', 'conv2_2', 'avgpool3_2', 'avgpool2_2'], \
            "the merging way is not exist!"
        self.cpe_per_satge = cpe_per_satge
        if merging_way == 'conv3_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=1),
                norm_layer(out_channels),
            )
        elif merging_way == 'conv2_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2), padding=0),
                norm_layer(out_channels),
            )
        elif merging_way == 'avgpool3_2':
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, padding=1),
                norm_layer(out_channels),
            )
        else:
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, padding=0),
                norm_layer(out_channels),
            )
        if self.cpe_per_satge:
            self.pos_embed = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1, groups=out_channels)

    def forward(self, x):
        # x: B, C, H ,W
        x = self.proj(x)
        if self.cpe_per_satge:
            x = x + self.pos_embed(x)
        return x


class PatchExpand(nn.Module):
    """ Patch Expanding Layer.
    """

    def __init__(self, in_channels, out_channels, merging_way, cpe_per_satge, norm_layer=nn.BatchNorm2d):
        super().__init__()
        assert merging_way in ['conv3_2', 'conv2_2', 'avgpool3_2', 'avgpool2_2'], \
            "the merging way is not exist!"
        self.cpe_per_satge = cpe_per_satge
        if merging_way == 'conv3_2':
            self.proj = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                   output_padding=(1, 1)),
                norm_layer(out_channels),
            )
        elif merging_way == 'conv2_2':
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2), padding=0),
                norm_layer(out_channels),
            )
        elif merging_way == 'avgpool3_2':
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, padding=1),
                norm_layer(out_channels),
            )
        else:
            self.proj = nn.Sequential(
                nn.AvgPool2d(in_channels, out_channels, padding=0),
                norm_layer(out_channels),
            )
        if self.cpe_per_satge:
            self.pos_embed = nn.Conv2d(out_channels, out_channels, (3, 3), padding=1, groups=out_channels)

    def forward(self, x):
        # x: B, C, H ,W
        x = self.proj(x)
        if self.cpe_per_satge:
            x = x + self.pos_embed(x)
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
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, self.output_dim)
        x = self.norm(x)

        return x


class LDstage(nn.Module):
    """ A basic Dilate Transformer layer for one stage.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None):
        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            LDBlock(dim=dim, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    dilation=dilation,
                    LD=0 if (i % 2 == 0) else 1,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_satge) if downsample else nn.Identity()

    def forward(self, i):
        for blk in self.blocks:
            i = blk(i)
        i = self.downsample(i)
        return i


class Globalstage(nn.Module):
    """ A basic Transformer layer for one stage."""

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_satge=False, cpe_per_block=False,
                 downsample=True, merging_way=None):
        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            GlobalBlock(dim=dim, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        self.downsample = PatchMerging(dim, int(dim * 2), merging_way, cpe_per_satge) if downsample else nn.Identity()

    def forward(self, i):
        for blk in self.blocks:
            i = blk(i)
        i = self.downsample(i)
        return i


class LDstage_up(nn.Module):
    """ A basic Dilate Transformer layer for one stage.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, dilation,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, cpe_per_satge=False, cpe_per_block=False,
                 upsample=True, merging_way=None):

        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            LDBlock(dim=dim, input_resolution=input_resolution,
                    num_heads=num_heads, window_size=window_size,
                    dilation=dilation,
                    LD=0 if (i % 2 == 0) else 1,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop, attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(dim, int(dim / 2), merging_way, cpe_per_satge) if upsample else nn.Identity()
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.upsample(x)
        return x


class Globalstage_up(nn.Module):
    """ A basic Transformer layer for one stage."""

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 cpe_per_satge=False, cpe_per_block=False,
                 upsample=True, merging_way=None):

        super().__init__()
        # build blocks
        self.blocks = nn.ModuleList([
            GlobalBlock(dim=dim, num_heads=num_heads,
                        mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                        qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer, act_layer=act_layer, cpe_per_block=cpe_per_block)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = PatchExpand(dim, int(dim / 2), merging_way, cpe_per_satge) if upsample else nn.Identity()
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.upsample(x)
        return x


class UDIT(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dim=96, dilation=[8, 4, 2, 1],
                 window_size=7,
                 depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0.1,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 merging_way='conv3_2',
                 patch_way='overlaping',
                 dilate_attention=[False, False, True, True],
                 downsamples=[True, True, True, False],
                 cpe_per_satge=False, cpe_per_block=True, final_upsample="expand_first"):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        norm_layer = norm_layer

        # patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, patch_way=patch_way)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [i.item() for i in torch.linspace(0, drop_path, sum(depths))]

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if dilate_attention[i_layer]:
                layer = LDstage(dim=int(embed_dim * 2 ** i_layer),
                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                  patches_resolution[1] // (2 ** i_layer)),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                dilation=dilation[i_layer],
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop, attn_drop=attn_drop,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=downsamples[i_layer],
                                cpe_per_block=cpe_per_block,
                                cpe_per_satge=cpe_per_satge,
                                merging_way=merging_way
                                )
            else:
                layer = Globalstage(dim=int(embed_dim * 2 ** i_layer),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=downsamples[i_layer],
                                    cpe_per_block=cpe_per_block,
                                    cpe_per_satge=cpe_per_satge,
                                    merging_way=merging_way
                                    )
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                              self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                       int((embed_dim * 2 ** (self.num_layers - 1 - i_layer)) / 2), merging_way,
                                       cpe_per_satge)
            else:
                if dilate_attention[self.num_layers - 1 - i_layer]:
                    layer_up = LDstage_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                          input_resolution=(
                                              patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                              patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                          depth=depths[(self.num_layers - 1 - i_layer)],
                                          num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                          window_size=window_size,
                                          dilation=dilation[(self.num_layers - 1 - i_layer)],
                                          mlp_ratio=self.mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          drop=drop, attn_drop=attn_drop,
                                          drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                              depths[:(self.num_layers - 1 - i_layer) + 1])],
                                          norm_layer=norm_layer,
                                          upsample=downsamples[i_layer],
                                          cpe_per_block=cpe_per_block,
                                          cpe_per_satge=cpe_per_satge,
                                          merging_way=merging_way
                                          )
                else:
                    layer_up = Globalstage_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                              input_resolution=(
                                                  patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                                  patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                              depth=depths[(self.num_layers - 1 - i_layer)],
                                              num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                              mlp_ratio=self.mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop, attn_drop=attn_drop,
                                              drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                                  depths[:(self.num_layers - 1 - i_layer) + 1])],
                                              norm_layer=norm_layer,
                                              upsample=downsamples[i_layer],
                                              cpe_per_block=cpe_per_block,
                                              cpe_per_satge=cpe_per_satge,
                                              merging_way=merging_way
                                              )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            self.up = FinalPatchExpand_X4(input_resolution=(img_size // patch_size, img_size // patch_size),
                                          dim_scale=4, dim=embed_dim)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=(1, 1),
                                    bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    # Encoder and Bottleneck
    def forward_features(self, i):
        i = self.patch_embed(i)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(i)
            i = layer(i)

        B, C, H, W = i.shape
        i = i.flatten(2).transpose(1, 2)
        i = self.norm(i)  # B L C
        i = i.view(B, C, H, W)

        return i, x_downsample

    # Decoder and Skip connection
    def forward_up_features(self, i, i_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                i = layer_up(i)
            else:
                i = torch.cat([i, i_downsample[3 - inx]], 1)
                i = i.permute(0, 2, 3, 1)
                i = self.concat_back_dim[inx](i)
                i = i.permute(0, 3, 1, 2)
                i = layer_up(i)

        i = i.flatten(2).transpose(1, 2)
        i = self.norm_up(i)  # B L C

        return i

    def up_x4(self, i):
        H, W = self.patches_resolution
        B, L, C = i.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == "expand_first":
            i = self.up(i)
            i = i.view(B, 4 * H, 4 * W, -1)
            i = i.permute(0, 3, 1, 2)  # B,C,H,W
            i = self.output(i)

        return i

    def forward(self, i):
        i, x_downsample = self.forward_features(i)
        i = self.forward_up_features(i, x_downsample)
        i = self.up_x4(i)

        return i


@register_model
def udit_tiny(**kwargs):
    model = UDIT(depths=[2, 2, 6, 2], embed_dim=72, num_heads=[3, 6, 12, 24], **kwargs)
    model.default_cfg = _cfg()
    return model


if __name__ == "__main__":
    x = torch.rand([2, 3, 224, 224])
    m = udit_tiny()
    y = m(x)
    print(y.shape)
