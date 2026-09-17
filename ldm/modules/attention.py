from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn as nn

import math
import torch.nn as nn
from sklearn.cluster import KMeans
from einops import rearrange

# from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
from torch.utils import checkpoint

import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("Xformers is not available. Install via ")


def exists(val):
    return val is not None


def uniq(arr):
    return {el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def fill_inf_from_mask(self, sim, mask):
        if mask is not None:
            B, M = mask.shape
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1).reshape(B * self.heads, 1, -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim

    def forward_plain(self, x, key, value, mask=None):
        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(key)  # B*M*(H*C)
        v = self.to_v(value)  # B*M*(H*C)

        B, N, HC = q.shape
        _, M, _ = key.shape
        H = self.heads
        C = HC // H

        q = q.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        k = k.view(B, M, H, C).permute(0, 2, 1, 3).reshape(B * H, M, C)  # (B*H)*M*C
        v = v.view(B, M, H, C).permute(0, 2, 1, 3).reshape(B * H, M, C)  # (B*H)*M*C

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale  # (B*H)*N*M
        self.fill_inf_from_mask(sim, mask)
        attn = sim.softmax(dim=-1)  # (B*H)*N*M

        out = torch.einsum('b i j, b j d -> b i d', attn, v)  # (B*H)*N*C
        out = out.view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, (H * C))  # B*N*(H*C)

        return self.to_out(out)

    def forward(self, x, key, value, mask=None):
        if not XFORMERS_IS_AVAILABLE:
            return self.forward_plain(x, key, value, mask)

        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(key)  # B*M*(H*C)
        v = self.to_v(value)  # B*M*(H*C)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0., interaction_scale=1.3):# Define interaction scaling coefficient
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        # Added a learnable interactive scaling factor parameter
        self.interaction_scale = interaction_scale

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward_plain(self, x):
        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(x)  # B*N*(H*C)
        v = self.to_v(x)  # B*N*(H*C)

        B, N, HC = q.shape
        H = self.heads
        C = HC // H

        q = q.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        k = k.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        v = v.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale * self.interaction_scale  # (B*H)*N*N
        #sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1)  # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v)  # (B*H)*N*C
        out = out.view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, (H * C))  # B*N*(H*C)

        return self.to_out(out)

    def forward(self, x, context=None, mask=None):
        if not XFORMERS_IS_AVAILABLE:
            return self.forward_plain(x)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)
        out = out * self.interaction_scale 

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class SelfAttention1(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def forward_plain(self, x):
        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(x)  # B*N*(H*C)
        v = self.to_v(x)  # B*N*(H*C)

        B, N, HC = q.shape
        H = self.heads
        C = HC // H

        q = q.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        k = k.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        v = v.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1)  # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v)  # (B*H)*N*C
        out = out.view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, (H * C))  # B*N*(H*C)

        return self.to_out(out)

    def forward(self, x, context=None, mask=None):
        if not XFORMERS_IS_AVAILABLE:
            return self.forward_plain(x)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class GatedCrossAttentionDense(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head):
        super().__init__()

        self.attn = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads,
                                   dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one
        self.scale = 1

    def forward(self, x, objs):
        x = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(self.norm1(x), objs, objs)
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x


class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one 
        self.scale = 1

    def forward(self, x, kgs):
        N_visual = x.shape[1]

        if kgs is None:
            device = x.device
            kgs = torch.zeros(1, 90, 768, device=device) 

        kgs = self.linear(kgs)

        x = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(self.norm1(torch.cat([x, kgs], dim=1)))[:,
                                                           0:N_visual, :]
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x

class GatedSelfAttentionDense2(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))
        self.scale = 1

    def forward(self, x, objs):
        B, N_visual, _ = x.shape
        B, N_ground, _ = objs.shape

        objs = self.linear(objs)

        # sanity check
        size_v = math.sqrt(N_visual)
        size_g = math.sqrt(N_ground)
        assert int(size_v) == size_v, "Visual tokens must be square rootable"
        assert int(size_g) == size_g, "Grounding tokens must be square rootable"
        size_v = int(size_v)
        size_g = int(size_g)

        # select grounding token and resize it to visual token size as residual
        out = self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, N_visual:, :]
        out = out.permute(0, 2, 1).reshape(B, -1, size_g, size_g)
        out = torch.nn.functional.interpolate(out, (size_v, size_v), mode='bicubic')
        residual = out.reshape(B, -1, N_visual).permute(0, 2, 1)

        # add residual to visual feature
        x = x + self.scale * torch.tanh(self.alpha_attn) * residual
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x
    
class MaskCrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
    def forward(self, x, objs, subject_mask, action_mask, object_mask):
        """
        :param x: query tensor of shape (B, N, query_dim)
        :param objs: Bond and value tensors of the shape (B, M, key_dim)
        :param subject_mask: Body mask, shaped like (B, hw, C)
        :param action_mask: Action mask in shape (B, hw, C)
        :param object_mask: Object mask of shape (B, hw, C)
        :return: output tensor in the shape (B, N, query_dim)
        """
        q = self.to_q(x)
        k = self.to_k(objs)
        v = self.to_v(objs)

        B, N, HC = q.shape
        _, M, _ = k.shape
        H = self.heads
        C = HC // H

        q = q.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)
        k = k.view(B, M, H, C).permute(0, 2, 1, 3).reshape(B * H, M, C)
        v = v.view(B, M, H, C).permute(0, 2, 1, 3).reshape(B * H, M, C)

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale #（B*H,N,M）

        def process_mask(mask, M):
            hw = mask.shape[1]
            mask = mask.unsqueeze(1) # (B,1,hw,C)
            mask = mask.repeat(1,H,1,1) # (B,H,hw,C)
            mask = mask.view(B*H, hw, C)
            return mask[:, :, None].repeat(1,1,M,1) # (B*H, N, M, C)

        subject_mask = process_mask(subject_mask)
        action_mask = process_mask(action_mask)
        object_mask = process_mask(object_mask)

        attn_subject = F.softmax(sim + subject_mask.sum(dim=-1), dim=-1)
        attn_action = F.softmax(sim + action_mask.sum(dim=-1), dim=-1)
        attn_object = F.softmax(sim + object_mask.sum(dim=-1), dim=-1)

        out_subject = torch.einsum('b i j, b j d -> b i d', attn_subject, v) * subject_mask
        out_action = torch.einsum('b i j, b j d -> b i d', attn_action, v) * action_mask
        out_object = torch.einsum('b i j, b j d -> b i d', attn_object, v) * object_mask

        out = out_subject + out_action + out_object

        out = out.view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, H*C)
        return self.to_out(out)
    
    
# Features-Enhanced Multi-scale Convolutional Network.
class MultiScaleMaskCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU()
        ) 
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        ) 
        self.conv_fuse = nn.Conv2d(in_channels*2, in_channels, kernel_size=1)
        
    def forward(self, x):
        B, HW, C = x.shape
        H = W = int(math.sqrt(HW))
        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2)
        
        out1 = self.branch1(x_img)
        out2 = self.branch2(x_img)
        out = torch.cat([out1, out2], dim=1)
        out = self.conv_fuse(out)
        
        return out.permute(0, 2, 3, 1).view(B, HW, C)

class BasicTransformerBlock(nn.Module):
    vis_counter = 0 

    def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=True):
        super().__init__()
        self.attn1 = SelfAttention1(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)
        self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads,
                                    dim_head=d_head)        
        self.subject_cnn = MultiScaleMaskCNN(query_dim)
        self.object_cnn = MultiScaleMaskCNN(query_dim)
        self.fusion_conv = nn.Conv1d(query_dim*2, query_dim, kernel_size=1)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)

        self.use_checkpoint = use_checkpoint

        self.linear_proj = nn.Linear(768, query_dim, bias=False)  

        self.gate_layer = nn.Sequential(
            nn.Linear(query_dim, query_dim // 2), 
            nn.ReLU(),
            nn.Linear(query_dim // 2, 1)       
        )

        self.gate_proj = nn.Sequential(
            nn.Linear(query_dim * 2, query_dim), 
            nn.ReLU(),
            nn.Linear(query_dim, query_dim),      
            nn.Sigmoid()                         
        )


        gate_heads = []
        for _ in range(4):  
            gate_heads.append(nn.Sequential(
                nn.Linear(query_dim * 2, query_dim),
                nn.ReLU(),
                nn.Linear(query_dim, query_dim),
                nn.Sigmoid()
            ))
        self.gate_heads = nn.ModuleList(gate_heads)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(query_dim * 2, query_dim * 2), 
            nn.GELU(),                                
            nn.Linear(query_dim * 2, query_dim),     
            nn.LayerNorm(query_dim),                  
            nn.Dropout(0.1)                           
        )

        if fuser_type == "gatedSA":
            # note key_dim here actually is context_dim
            self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head)
        elif fuser_type == "gatedSA2":
            # note key_dim here actually is context_dim
            self.fuser = GatedSelfAttentionDense2(query_dim, key_dim, n_heads, d_head)
        elif fuser_type == "gatedCA":
            self.fuser = GatedCrossAttentionDense(query_dim, key_dim, value_dim, n_heads, d_head)
        else:
            assert False

    def visualize_tensors(self, subject_mask, object_mask, x_subject, x_object):
        B, hw, C = x_subject.shape
        H = W = int(math.sqrt(hw))
        
        # reshape to image format
        def to_img(x):
            return x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        subject_mask_img = to_img(subject_mask)[:, :3]  
        object_mask_img = to_img(object_mask)[:, :3]
        x_subject_img = to_img(x_subject)[:, :3].clamp(0, 1)
        x_object_img = to_img(x_object)[:, :3].clamp(0, 1)

        grid = make_grid(torch.cat([
            subject_mask_img[0:1],
            object_mask_img[0:1],
            x_subject_img[0:1],
            x_object_img[0:1]
        ], dim=0), nrow=4, padding=5, normalize=True)

        plt.figure(figsize=(8, 2))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")

        save_dir = "visualizations"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filename = os.path.join(save_dir, f"visualization_{BasicTransformerBlock.vis_counter:04d}.png")
        BasicTransformerBlock.vis_counter += 1 
        plt.savefig(filename)
        plt.close()
        print(f"Image saved at: {filename}")

    def visualize_features(self, x_subject, x_object, x_sub_cnn, x_obj_cnn):
        B, hw, C = x_subject.shape
        H = W = int(math.sqrt(hw))
        
        def to_img(x):
            return x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        x_subject_img = to_img(x_subject)[:, :3].clamp(0, 1)
        x_object_img = to_img(x_object)[:, :3].clamp(0, 1)
        x_sub_cnn_img = to_img(x_sub_cnn)[:, :3].clamp(0, 1)
        x_obj_cnn_img = to_img(x_obj_cnn)[:, :3].clamp(0, 1)

        grid = make_grid(torch.cat([
            x_subject_img[0:1],    
            x_object_img[0:1],    
            x_sub_cnn_img[0:1],    
            x_obj_cnn_img[0:1],   
        ], dim=0), nrow=2, normalize=True)

        plt.figure(figsize=(8, 4))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        
        save_dir = "visualizations2"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"features_{BasicTransformerBlock.vis_counter:04d}.png")
        BasicTransformerBlock.vis_counter += 1
        plt.savefig(filename)
        plt.close()

    def visualize_x_and_xc(self, x, xc):
        B, hw, C = x.shape
        H = W = int(math.sqrt(hw))
        
        def to_img(tensor):
            return tensor.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        x_img = to_img(x)[:, :3].clamp(0, 1)
        xc_img = to_img(xc)[:, :3].clamp(0, 1)

        grid = make_grid(torch.cat([
            x_img[0:1],    
            xc_img[0:1],  
        ], dim=0), nrow=2, normalize=True)

        plt.figure(figsize=(6, 3))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        
        save_dir = "visualizations_x_xc"
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"x_xc_{BasicTransformerBlock.vis_counter:04d}.png")
        BasicTransformerBlock.vis_counter += 1
        plt.savefig(filename)
        plt.close()

# Soft mask gengeration 
    def generate_masks_with_semantics(self, x, subject_positive_embeddings, object_positive_embeddings):
        """
        Use semantically guided clustering methods to generate body and object masks
        :param x: Input eigentensor (B, hw, C)
        :param subject_positive_embeddings: Subject Semantic Embeddings [B,N,768]
        :param object_positive_embeddings: Object Semantic Embedding[B,N,768]
        :return: Subject mask and object mask (B, hw, 1)
        """
        B, hw, C = x.shape
        print(f"x.shape={x.shape}")
        B,N,_ = subject_positive_embeddings.shape
        H = W = int(math.sqrt(hw))
        assert H * W == hw, f"hw={hw}, H={H}, W={W}, H*W={H*W}" 

        if subject_positive_embeddings.shape[-1] != C:
            subject_positive_embeddings = self.linear_proj(subject_positive_embeddings)    
        if object_positive_embeddings.shape[-1] != C:
            object_positive_embeddings = self.linear_proj(object_positive_embeddings)  

        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        subject_gate = torch.sigmoid(self.gate_layer(subject_positive_embeddings))
        object_gate = torch.sigmoid(self.gate_layer(object_positive_embeddings))  
        subject_attention = torch.einsum('bnc,bchw->bhw', subject_gate, x_img)  
        object_attention = torch.einsum('bnc,bchw->bhw', object_gate, x_img) 
        
        # Define the temperature scaling coefficient
        temperature = 2.0 

        subject_attention = F.softmax(subject_attention.view(B, -1) / temperature, dim=-1).view(B, H, W)
        object_attention = F.softmax(object_attention.view(B, -1) / temperature, dim=-1).view(B, H, W)

        subject_mask = subject_attention.unsqueeze(-1).view(B, -1, 1)  # (B, hw, 1)
        object_mask = object_attention.unsqueeze(-1).view(B, -1, 1)    # (B, hw, 1)

        return subject_mask, object_mask, subject_attention, object_attention


    def forward(self, x, context, objs, objs1, kgs, subject_boxes, action_boxes, object_boxes, subject_positive_embeddings, object_positive_embeddings, timesteps):
        if self.use_checkpoint and x.requires_grad:
            return checkpoint.checkpoint(self._forward, x, context, objs, objs1, kgs, subject_boxes, action_boxes, object_boxes, subject_positive_embeddings, object_positive_embeddings, timesteps)
        else:
            return self._forward(x, context, objs, objs1, kgs, subject_boxes, action_boxes, object_boxes, subject_positive_embeddings, object_positive_embeddings, timesteps)

    def _forward(self, x, context, objs, objs1, kgs, subject_boxes, action_boxes, object_boxes, subject_positive_embeddings, object_positive_embeddings, timesteps):

        x = self.attn1(self.norm1(x)) + x
        print(f'x shape: {x.shape}')
        if timesteps is not None and timesteps[0] >= 681: # Sampling control
            x1 = self.fuser(x, objs1)  
            print(f"objs shape: {objs.shape}") 
        else:
            x1 = self.fuser(x, objs) 
        print(f'timesteps: {timesteps}')
        print(f'timesteps: {timesteps[0]}')

        print(f'x1 shape: {x1.shape}')

        if kgs is None:
            kgs = torch.zeros(1, 90, 768, device=x.device)

        x2 = self.fuser(x, kgs)
        print(f"kgs shape: {kgs.shape}") 
        print(f"x2 shape: {x2.shape}")
        x = 0.7*x1 + 0.3*x2 # w is fusion weight:w=0.7

        #  --------------Dynamic gating integration ------------------
        # gate = torch.sigmoid(self.gate_proj(torch.cat([x1, x2], dim=-1)))
        # x = gate * x1 + (1 - gate) * x2 

        print(f'x shape: {x.shape}')
        print(f"subject_positive_embeddings shape: {subject_positive_embeddings.shape}") # [1,30,768]
        print(f"object_positive_embeddings shape: {object_positive_embeddings.shape}")

        B, hw, C = x.shape
        h = w = int(math.sqrt(hw))
        
        # Entity control network
        subject_mask, object_mask, subject_attention, object_attention = self.generate_masks_with_semantics(x, subject_positive_embeddings, object_positive_embeddings)  # (B, hw, 1)

        # Multiply the soft attention mask with the feature map to retain information about regions of interest and suppress irrelevant regions
        subject_mask = subject_mask.expand(-1, -1, C)       # (B, hw, C)
        object_mask = object_mask.expand(-1, -1, C)         # (B, hw, C)
  
        x_subject = x * subject_mask  
        x_object = x * object_mask    
        
        # CNN
        x_sub_cnn = self.subject_cnn(x_subject)   # (B, hw, C)
        x_obj_cnn = self.object_cnn(x_object)     # (B, hw, C)
    
        # Fusion
        # #combined = torch.cat([x_subject, x_object], dim=-1) 
        combined = torch.cat([x_sub_cnn, x_obj_cnn], dim=-1) 
        combined = combined.transpose(1, 2)  # (B, 2*C, hw)
        xc = self.fusion_conv(combined)      # (B, C, hw) 
        xc = xc.transpose(1, 2)              # (B, hw, C) 
        print(f"xc shape: {xc.shape}")   

        # x_full = torch.cat([xc, x], dim=-1)  # xc: Prospect integration features; x: Original features
        # x = self.attn2(self.norm2(x_full), context, context) + x
        #x = xc + x

        # Cross-attention for action correction
        x = self.attn2(self.norm2(x+xc), context, context) + x
        print(f'x_cross_shape: {x.shape}')
        x = self.ff(self.norm3(x)) + x
        print(f'x_finalshape: {x.shape}')

        return x 

class SpatialTransformer(nn.Module):
    def __init__(self, in_channels, key_dim, value_dim, n_heads, d_head, depth=1, fuser_type=None, use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        query_dim = n_heads * d_head
        #query_dim = in_channels
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 query_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(query_dim, key_dim, value_dim, n_heads, d_head, fuser_type,
                                   use_checkpoint=use_checkpoint)
             for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(query_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def _forward(self, x, context, objs, objs1, kgs, subject_boxes, action_boxes, object_boxes, subject_positive_embeddings, object_positive_embeddings, timesteps):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        for block in self.transformer_blocks:
            x = block(x, context, objs, objs1, kgs, subject_boxes, action_boxes, object_boxes, subject_positive_embeddings, object_positive_embeddings, timesteps)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in 


class HOISpatialTransformer(nn.Module):
    def __init__(self, in_channels, key_dim, value_dim, n_heads, d_head, depth=1, fuser_type=None, use_checkpoint=True):
        super().__init__()
        self.in_channels = in_channels
        query_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 query_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(query_dim, key_dim, value_dim, n_heads, d_head, fuser_type,
                                   use_checkpoint=use_checkpoint)
             for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(query_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context, objs, objs1, kgs, subject_boxes, action_boxes, object_boxes, subject_positive_embeddings, object_positive_embeddings, timesteps):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')  # .contiguous()

        for block in self.transformer_blocks:
            x = block(x, context, objs, objs1, kgs, subject_boxes, action_boxes, object_boxes, subject_positive_embeddings, object_positive_embeddings, timesteps)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        x = self.proj_out(x)
        return x + x_in
