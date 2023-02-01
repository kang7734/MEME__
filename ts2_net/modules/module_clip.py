"""
Adapted from: https://github.com/openai/CLIP/blob/main/clip/clip.py
"""
from collections import OrderedDict
from typing import Tuple, Union

import hashlib
import os
import urllib
import warnings
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}
_PT_NAME = {
    "RN50": "RN50.pt",
    "RN101": "RN101.pt",
    "RN50x4": "RN50x4.pt",
    "RN50x16": "RN50x16.pt",
    "ViT-B/32": "ViT-B-32.pt",
    "ViT-B/16": "ViT-B-16.pt",
}

def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

# =============================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, layer: int, attn_mask=None):
        super().__init__()
        self.layer = layer

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple:tuple):
        x, video_frame = x_tuple

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return (x, video_frame)

    
class ResidualAttentionBlock_Multi(nn.Module):
    def __init__(self, d_model: int, n_head: int, layer: int, attn_mask=None):
        super().__init__()
        self.layer = layer

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple:tuple):
        x, video_frame = x_tuple

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return (x, video_frame)


class ResidualAttentionBlock_coarse(nn.Module):
    def __init__(self, d_model: int, n_head: int, layer: int, attn_mask=None):
        super().__init__()
        self.layer = layer
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple:tuple):
        x, video_frame = x_tuple

        ############Diff############
        if video_frame >= 1 and self.layer==1:
            P,Bt,D = x.shape
            new_list = []
            patches_origin = x[1:,:,:]
            patches_origin = patches_origin.reshape(Bt//12,(P-1)*12, D)
            #patches = patches_origin.detach()
            patches = patches_origin
            patch_front = patches.clone().detach().to(device=x.device)
            patch_front[patch_front!=patch_front] = 0.
            patch_front = patch_front / (patch_front.norm(dim=-1, keepdim=True) + 0.000000001)
            patch_back = patches.clone().detach().to(device=x.device)
            patch_back[patch_back!=patch_back] = 0.
            patch_back = patch_back / (patch_back.norm(dim=-1, keepdim=True) + 0.000000001)
            patch_back = patch_back.permute(0,2,1)

            cluster_sim_matrix = patch_front @ patch_back
            patch_sim_matrix = patch_front @ patch_back

            cluster_sim_matrix[cluster_sim_matrix <= 0.95] = 0.
            cluster_sim_matrix[cluster_sim_matrix > 0.95] = 1.
            #cluster_sim_matrix = ~torch.isnan(cluster_sim_matrix)
            cluster_sim_matrix = cluster_sim_matrix.type(torch.float16)

            patch_sim_matrix[patch_sim_matrix <= 0.95] = 0.
            patch_sim_matrix[patch_sim_matrix > 0.95] = 1.
            #patch_sim_matrix = ~torch.isnan(patch_sim_matrix)
            patch_sim_matrix = patch_sim_matrix.type(torch.float16)

            for i in range(len(cluster_sim_matrix)):
                matrix_i = torch.eye(len(cluster_sim_matrix[i])).to(device=x.device)
                clustering_matrix = torch.zeros_like(cluster_sim_matrix[i]).to(device=x.device)
                clustering_matrix = clustering_matrix + matrix_i

                cluster_num = 10
                cnt = 0
                while cluster_num >= 1:
                    cnt += 1
                    nonzero_count = torch.count_nonzero(cluster_sim_matrix[i], dim=1).to(device=x.device)
                    _, nonzero_count_idx = torch.max(nonzero_count, dim=0)
                    hub_node_idx = torch.arange(len(cluster_sim_matrix[i][nonzero_count_idx])).to(device=x.device) + 1
                    hub_node_idx = cluster_sim_matrix[i][nonzero_count_idx] * hub_node_idx
                    hub_node_idx = torch.nonzero(hub_node_idx).T
                    hub_node_idx = hub_node_idx.type(torch.long)

                    cluster_sim_matrix[i][hub_node_idx] = 0
                    #cluster_sub_matrix = cluster_sim_matrix[i][hub_node_idx].T
                    #cluster_sub_matrix[nonzero_count_idx] += 1
                    cluster_sim_matrix[i][hub_node_idx,nonzero_count_idx] = 1

                    clustering_matrix[hub_node_idx] = 0
                    clustering_matrix[hub_node_idx,nonzero_count_idx] = 1

                    patch_sim_matrix[i][hub_node_idx] = 0.

                    cluster_num = patch_sim_matrix[i] - matrix_i
                    cluster_num[cluster_num == -1] = 0.
                    cluster_num = int(torch.sum(cluster_num))
                    if cnt >= 1000:
                        break

                clustering_matrix = clustering_matrix.type(torch.float16)
                patches_origin = patches_origin.type(torch.float16)
                clustering_matrix = clustering_matrix.detach()
                new = clustering_matrix @ patches_origin[i]
                new = new.type(torch.float16)
                new_list.append(new)

            new_data_fin = torch.stack(new_list, dim=0)
            new_data_fin = new_data_fin.reshape(P-1,Bt,D)
            new_data_fin = new_data_fin.type(torch.float16)
            x = torch.cat((x[:1,:,:], new_data_fin))
            x = x.type(torch.float16)
            ############Diff############

            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        #x = x + self.attention(self.ln_1(x))
        #x = x + self.mlp(self.ln_2(x))
        return (x, video_frame)



class ResidualAttentionBlock_multi_coarse(nn.Module):
    def __init__(self, d_model: int, n_head: int, layer: int, attn_mask=None):
        super().__init__()
        self.layer = layer
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        attn_mask_ = self.attn_mask
        if self.attn_mask is not None and hasattr(self.attn_mask, '__call__'):
            attn_mask_ = self.attn_mask(x.size(0))   # LND

        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask_)[0]

    def forward(self, x_tuple:tuple):
        x, video_frame = x_tuple

        ############Diff############
        if video_frame >= 1 and self.layer==1:
            P,Bt,D = x.shape
            new_list = []
            patches_origin = x[1:,:,:]
            patches_origin = patches_origin.reshape(Bt//12,(P-1)*12, D)
            #patches = patches_origin.detach()
            patches = patches_origin
            patch_front = patches.clone().detach().to(device=x.device)
            patch_front[patch_front!=patch_front] = 0.
            patch_front = patch_front / (patch_front.norm(dim=-1, keepdim=True) + 0.000000001)
            patch_back = patches.clone().detach().to(device=x.device)
            patch_back[patch_back!=patch_back] = 0.
            patch_back = patch_back / (patch_back.norm(dim=-1, keepdim=True) + 0.000000001)
            patch_back = patch_back.permute(0,2,1)

            cluster_sim_matrix = patch_front @ patch_back
            patch_sim_matrix = patch_front @ patch_back

            cluster_sim_matrix[cluster_sim_matrix <= 0.95] = 0.
            cluster_sim_matrix[cluster_sim_matrix > 0.95] = 1.
            #cluster_sim_matrix = ~torch.isnan(cluster_sim_matrix)
            cluster_sim_matrix = cluster_sim_matrix.type(torch.float16)

            patch_sim_matrix[patch_sim_matrix <= 0.95] = 0.
            patch_sim_matrix[patch_sim_matrix > 0.95] = 1.
            #patch_sim_matrix = ~torch.isnan(patch_sim_matrix)
            patch_sim_matrix = patch_sim_matrix.type(torch.float16)

            for i in range(len(cluster_sim_matrix)):
                matrix_i = torch.eye(len(cluster_sim_matrix[i])).to(device=x.device)
                clustering_matrix = torch.zeros_like(cluster_sim_matrix[i]).to(device=x.device)
                clustering_matrix = clustering_matrix + matrix_i

                cluster_num = 10
                cnt = 0
                while cluster_num >= 1:
                    cnt += 1
                    nonzero_count = torch.count_nonzero(cluster_sim_matrix[i], dim=1).to(device=x.device)
                    _, nonzero_count_idx = torch.max(nonzero_count, dim=0)
                    hub_node_idx = torch.arange(len(cluster_sim_matrix[i][nonzero_count_idx])).to(device=x.device) + 1
                    hub_node_idx = cluster_sim_matrix[i][nonzero_count_idx] * hub_node_idx
                    hub_node_idx = torch.nonzero(hub_node_idx).T
                    hub_node_idx = hub_node_idx.type(torch.long)

                    cluster_sim_matrix[i][hub_node_idx] = 0
                    #cluster_sub_matrix = cluster_sim_matrix[i][hub_node_idx].T
                    #cluster_sub_matrix[nonzero_count_idx] += 1
                    cluster_sim_matrix[i][hub_node_idx,nonzero_count_idx] = 1

                    clustering_matrix[hub_node_idx] = 0
                    clustering_matrix[hub_node_idx,nonzero_count_idx] = 1

                    patch_sim_matrix[i][hub_node_idx] = 0.

                    cluster_num = patch_sim_matrix[i] - matrix_i
                    cluster_num[cluster_num == -1] = 0.
                    cluster_num = int(torch.sum(cluster_num))
                    if cnt >= 1000:
                        break

                clustering_matrix = clustering_matrix.type(torch.float16)
                patches_origin = patches_origin.type(torch.float16)
                clustering_matrix = clustering_matrix.detach()
                new = clustering_matrix @ patches_origin[i]
                new = new.type(torch.float16)
                new_list.append(new)

            new_data_fin = torch.stack(new_list, dim=0)
            new_data_fin = new_data_fin.reshape(P-1,Bt,D)
            new_data_fin = new_data_fin.type(torch.float16)
            x = torch.cat((x[:1,:,:], new_data_fin))
            x = x.type(torch.float16)
            ############Diff############

            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        #x = x + self.attention(self.ln_1(x))
        #x = x + self.mlp(self.ln_2(x))
        return (x, video_frame)




class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, layer, attn_mask) for layer in range(layers)])

    def forward(self, x: torch.Tensor, video_frame=-1):
        return self.resblocks((x, video_frame))[0]

class Transformer_Coarse(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_coarse(width, heads, layer, attn_mask) for layer in range(layers)])

    def forward(self, x: torch.Tensor, video_frame=-1):
        return self.resblocks((x, video_frame))[0]

class Transformer_multi(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, layer, attn_mask) for layer in range(layers)])

    def forward(self, x: torch.Tensor, video_frame=-1):
        return self.resblocks((x, video_frame))[0]

class Transformer_Coarse_multi(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock_coarse(width, heads, layer, attn_mask) for layer in range(layers)])

    def forward(self, x: torch.Tensor, video_frame=-1):
        return self.resblocks((x, video_frame))[0]
        




class VisualTransformer_Coarse(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 linear_patch: str = '2d',):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer_Coarse(width, layers, heads)        
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)

    def forward(self, x: torch.Tensor, video_frame=-1):

        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            x_3d = self.conv2(x_3d)     # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)      # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, video_frame=video_frame)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Move the three lines below to `encode_image` for entire hidden sequence
        # x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj

        return x


class VisualTransformer_Coarse_multi(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 linear_patch: str = '2d',):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer_Coarse(width, layers, heads)        
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)

    def forward(self, x: torch.Tensor, video_frame=-1):

        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            x_3d = self.conv2(x_3d)     # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)      # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, video_frame=video_frame)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Move the three lines below to `encode_image` for entire hidden sequence
        # x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj

        return x




class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 linear_patch: str = '2d',):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)        
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)

    def forward(self, x: torch.Tensor, video_frame=-1):

        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            x_3d = self.conv2(x_3d)     # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)      # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, video_frame=video_frame)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Move the three lines below to `encode_image` for entire hidden sequence
        # x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj

        return x




class VisualTransformer_multi(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 linear_patch: str = '2d',):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)        
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # For 3D
        assert linear_patch in ['2d', '3d']
        self.linear_patch = linear_patch
        if self.linear_patch == '3d':
            self.conv2 = nn.Conv3d(in_channels=3, out_channels=width, kernel_size=(3, patch_size, patch_size),
                                   stride=(1, patch_size, patch_size), padding=(1, 0, 0), bias=False)

    def forward(self, x: torch.Tensor, video_frame=-1):

        if self.linear_patch == '3d':
            assert video_frame != -1
            x_3d = x.reshape(-1, video_frame, x.shape[-3], x.shape[-2], x.shape[-1])
            x_3d = x_3d.permute(0, 2, 1, 3, 4)
            x_3d = self.conv2(x_3d)     # shape = [*, width, frame, grid, grid]
            x_3d = x_3d.permute(0, 2, 1, 3, 4)      # shape = [*, frame, width, grid, grid]
            x = x_3d.reshape(-1, x_3d.shape[-3], x_3d.shape[-2], x_3d.shape[-1]).contiguous() # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, video_frame=video_frame)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Move the three lines below to `encode_image` for entire hidden sequence
        # x = self.ln_post(x[:, 0, :])
        # if self.proj is not None:
        #     x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # vision linear of patch
                 linear_patch: str = '2d',
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch
            )
            self.visual_Coarse = VisualTransformer_Coarse(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch
            )
            self.visual_multi = VisualTransformer_multi(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch
            )
            self.visual_Coarse_multi = VisualTransformer_Coarse_multi(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                linear_patch=linear_patch
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )
        self.transformer_Coarse = Transformer_Coarse(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )

        self.transformer_multi = Transformer_multi(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )
        self.transformer_Coarse_multi = Transformer_Coarse_multi(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask
        )


        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.token_embedding_ = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.positional_embedding_ = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.ln_final_ = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.text_projection_ = nn.Parameter(torch.empty(transformer_width, embed_dim))

        self.logit_scale = nn.Parameter(torch.ones([]))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        for block in self.transformer_Coarse.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        for block in self.transformer_multi.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        for block in self.transformer_Coarse_multi.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

        if self.text_projection_ is not None:
            nn.init.normal_(self.text_projection_, std=self.transformer_Coarse.width ** -0.5)
    @staticmethod
    def get_config(pretrained_clip_name="ViT-B/32"):
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ViT-B-32.pt")
        if pretrained_clip_name in _MODELS and pretrained_clip_name in _PT_NAME:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[pretrained_clip_name])
            
        if pretrained_clip_name in ["ViT-B/32", "ViT-B/16"] and os.path.exists(model_path):
            pass
        else:
            if pretrained_clip_name in _MODELS:
                model_path = _download(_MODELS[pretrained_clip_name])
            elif os.path.isfile(pretrained_clip_name):
                model_path = pretrained_clip_name
            else:
                raise RuntimeError(f"Model {pretrained_clip_name} not found; available models = {available_models()}")

        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        return state_dict

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.zeros(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, return_hidden=False, video_frame=-1):
        # with torch.no_grad():
        hidden = self.visual(image.type(self.dtype), video_frame=video_frame)
        hidden = self.visual.ln_post(hidden) @ self.visual.proj
        hidden_Coarse = self.visual_Coarse(image.type(self.dtype), video_frame=video_frame)
        hidden_Coarse = self.visual_Coarse.ln_post(hidden_Coarse) @ self.visual_Coarse.proj
        hidden_multi = self.visual_multi(image.type(self.dtype), video_frame=video_frame)
        hidden_multi = self.visual_multi.ln_post(hidden_multi) @ self.visual_multi.proj
        hidden_Coarse_multi = self.visual_Coarse_multi(image.type(self.dtype), video_frame=video_frame)
        hidden_Coarse_multi = self.visual_Coarse_multi.ln_post(hidden_Coarse_multi) @ self.visual_Coarse_multi.proj

        return hidden,hidden_Coarse,hidden_multi,hidden_Coarse_multi

    def encode_text(self, text, return_hidden=False):
        # with torch.no_grad():
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x_coarse = self.token_embedding_(text).type(torch.float16)  # [batch_size, n_ctx, d_model]

        pos_emd_ = self.positional_embedding_[:x_coarse.size(1), :].type(self.dtype)
        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)

        x_coarse = x_coarse + pos_emd_
        x = x + pos_emd

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x_coarse = x_coarse.permute(1, 0, 2)  # NLD -> LND
        x_coarse = self.transformer_Coarse(x_coarse)
        x_coarse = x_coarse.permute(1, 0, 2)  # LND -> NLD

        hidden = self.ln_final(x).type(self.dtype) @ self.text_projection
        hidden_coarse = self.ln_final_(x_coarse).type(self.dtype) @ self.text_projection_

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]
        x_coarse = hidden_coarse[torch.arange(hidden_coarse.shape[0]), text.argmax(dim=-1)]

        return x,x_coarse


    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

        for name in ["text_projection_", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
