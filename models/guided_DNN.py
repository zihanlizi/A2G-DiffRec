import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math



class AdaptiveAutoGuideNet(nn.Module):
    def __init__(self, item_dim: int, hidden: int = 32, w_max: float = 3.5, 
                 tau: float = 2.0, use_stats: bool = True, use_item_groups: bool = True, bonus_coefficient: float = 0.2,
                 use_stats_l1: bool = True, use_stats_entropy: bool = True, use_stats_norm_ratio: bool = True):
        super().__init__()
        assert w_max >= 1.0, "w_max must be >= 1.0"
        
        self.item_dim = int(item_dim)
        self.hidden = int(hidden)
        self.w_max = float(w_max)
        self.tau = float(tau) 
        self.use_stats = bool(use_stats)
        self.use_item_groups = bool(use_item_groups)
        self.bonus_coefficient = float(bonus_coefficient)
        # Ablation: control each statistical feature independently (only when use_stats=True)
        self.use_stats_l1 = self.use_stats and use_stats_l1
        self.use_stats_entropy = self.use_stats and use_stats_entropy
        self.use_stats_norm_ratio = self.use_stats and use_stats_norm_ratio

        base_feat = self.item_dim * 2  # concat(z1, z0)
        stat_feat = (1 if self.use_stats_l1 else 0) + (1 if self.use_stats_entropy else 0) + (1 if self.use_stats_norm_ratio else 0)
        self.in_dims = base_feat + stat_feat
        self.out_dims = 1

        self.mlp = nn.Sequential(
            nn.Linear(self.in_dims, self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.out_dims),
        )

        self._init_weights()
        
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _entropy(self, pred: torch.Tensor, dim: int = -1) -> torch.Tensor:
        p = F.softmax(pred / self.tau, dim=dim) 
        p = p.clamp_min(1e-12)  
        ent = -(p * p.log()).sum(dim=dim, keepdim=True)
        return ent

    def forward(self, z1: torch.Tensor, z0: torch.Tensor, 
                item_groups = None) -> torch.Tensor:
        assert z1.shape == z0.shape, "z1 and z0 must have the same shape"
        feats = [z1, z0]
        
        stat_feats = []
        if self.use_stats_l1:
            l1 = (z1 - z0).abs().mean(dim=1, keepdim=True)  # [B, 1] - L1 difference
            stat_feats.append(l1)
        if self.use_stats_entropy:
            H1 = self._entropy(z1)  # [B, 1] 
            H0 = self._entropy(z0)  # [B, 1] 
            dH = H1 - H0  # [B, 1] - Entropy difference
            stat_feats.append(dH)
        if self.use_stats_norm_ratio:
            z1_norm = z1.norm(p=2, dim=1, keepdim=True) + 1e-8
            z0_norm = z0.norm(p=2, dim=1, keepdim=True) + 1e-8
            norm_ratio = z1_norm / z0_norm  # [B, 1] - Norm ratio
            stat_feats.append(norm_ratio)
        feats.extend(stat_feats)

        h = torch.cat(feats, dim=1)  # [B, in_dims]
        
        g = self.mlp(h).squeeze(-1)  # [B]
        
        w_scaled = 1.0 + (self.w_max - 1.0) * torch.sigmoid(g)  # [B]
        
        if self.use_item_groups and item_groups is not None:
            if isinstance(item_groups, torch.Tensor):
                item_groups_tensor = item_groups.to(z1.device)
            elif isinstance(item_groups, np.ndarray):
                item_groups_tensor = torch.from_numpy(item_groups).to(z1.device)
            else:
                item_groups_tensor = torch.tensor(item_groups, device=z1.device)
            
            long_tail_mask = (item_groups_tensor == 2).float()  # [I]
            
            long_tail_score = (z1 * long_tail_mask).sum(dim=1)  # [B]
            
            bonus = torch.sigmoid(long_tail_score * 5.0)  # [B]
            w_scaled = w_scaled * (1.0 + self.bonus_coefficient * bonus)
            
            w_scaled = torch.clamp(w_scaled, min=1.0, max=self.w_max)
        
        return w_scaled  # [B]


class DNN(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat",
                 norm=False, dropout=0.5, cond_dim: int = 0):
        super(DNN, self).__init__()
        self.in_dims = list(in_dims)
        self.out_dims = list(out_dims)
        assert self.out_dims[0] == self.in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = int(emb_size)
        self.norm = norm
        self.cond_dim = int(cond_dim)

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        if self.time_type == "cat":
            first_in = self.in_dims[0] + self.time_emb_dim + self.cond_dim
            in_dims_temp = [first_in] + self.in_dims[1:]
        else:
            raise ValueError(f"Unimplemented timestep embedding type {self.time_type}")

        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([
            nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])
        ])
        self.out_layers = nn.ModuleList([
            nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])
        ])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in list(self.in_layers) + list(self.out_layers) + [self.emb_layer]:
            size = layer.weight.size()
            fan_out, fan_in = size[0], size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            nn.init.normal_(layer.bias, mean=0.0, std=0.001)


    def _mlp(self, h: torch.Tensor) -> torch.Tensor:
            for i, layer in enumerate(self.in_layers):
                h = torch.tanh(layer(h))
            for i, layer in enumerate(self.out_layers):
                h = layer(h)
                if i != len(self.out_layers) - 1:
                    h = torch.tanh(h)
            return h

    def forward(self, x, timesteps, cond: torch.Tensor = None):
        time_emb = timestep_embedding(timesteps, self.time_emb_dim).to(x.device)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)

        if cond is None:
            if self.cond_dim > 0:
                cond = torch.zeros(x.size(0), self.cond_dim, device=x.device, dtype=x.dtype)
            else:
                h = torch.cat([x, emb], dim=-1)
                return self._mlp(h)

        else:
            if self.cond_dim ==0:
                h = torch.cat([x, emb], dim=-1)
                return self._mlp(h)
            cond = cond.to(x.device, dtype=x.dtype)
            assert cond.dim()==2 and cond.size(0)==x.size(0) and cond.size(1)==self.cond_dim, \
                f"cond expected [B,{self.cond_dim}], got {list(cond.shape)}"

        h = torch.cat([x, emb, cond], dim=-1)
        return self._mlp(h)



class SelfGuidanceNet(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size=10, norm=False, dropout=0.5):
        super(SelfGuidanceNet, self).__init__()
        self.norm = norm

        # input: [x_t; y] -> output: residual
        in_dims_temp = [in_dims[0] * 2] + in_dims[1:]
        out_dims_temp = out_dims

        self.layers = nn.ModuleList([
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])
        ])

        self.out_layers = nn.ModuleList([
            nn.Linear(d_in, d_out)
            for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])
        ])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in list(self.layers) + list(self.out_layers):
            size = layer.weight.size()
            fan_out, fan_in = size[0], size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x_t, y):
        if self.norm:
            x_t = F.normalize(x_t)
        x_t = self.drop(x_t)

        h = torch.cat([x_t, y], dim=-1)

        for i, layer in enumerate(self.layers):
            h = layer(h)
            h = torch.tanh(h)

        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)

        return h

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding