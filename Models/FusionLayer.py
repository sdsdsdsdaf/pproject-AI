import torch
import torch.nn as nn
import torch.nn.functional as F


# 2. Fusion 모듈
class ConcatFusion(nn.Module):
    """기본 Late Fusion: 모든 모달 특징을 단순 연결(concat) 후 MLP"""
    def __init__(self, total_dim: int, fusion_dim: int, act_fn=nn.GELU):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            act_fn(),
        )

    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        if not feat_list:
            raise ValueError("No features to fuse")
        return self.fusion(torch.cat(feat_list, dim=1))
    
class AdvancedMLPFusion(nn.Module):
    """
    Concat Fusion 개선판: 깊은 MLP + LayerNorm + Dropout (2->3층)
    과적합 방지 + 표현력 강화
    """
    def __init__(self, total_dim, fusion_dim, dropout=0.1, act_fn=nn.GELU):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            act_fn(),
        )
    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        if not feat_list:
            raise ValueError("No features to fuse")
        return self.fusion(torch.cat(feat_list, dim=1))


class ModalGatingFusion(nn.Module):
    """Softmax 기반 모달 가중합: 각 모달에 중요도 할당 (sum_i=1)"""
    def __init__(self, feat_dims, fusion_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(sum(feat_dims), len(feat_dims)),
            nn.Softmax(dim=-1)
        )
        self.fusion = nn.Linear(sum(feat_dims), fusion_dim)
    
    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        if not feat_list:
            raise ValueError("No features to fuse")
        concat = torch.cat(feat_list, dim=1)
        weights = self.gate(concat)  # (B, num_modals)
        weighted = sum(w.unsqueeze(-1) * f for w, f in zip(weights.split(1,dim=1), feat_list))
        return self.fusion(weighted)


class CrossModalAttention(nn.Module):
    def __init__(self, feat_dims, total_dim, fusion_dim, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=total_dim//len(feat_dims),  # per modality dim
            num_heads=num_heads,
            batch_first=True
        )
        self.proj = nn.Linear(total_dim//len(feat_dims), fusion_dim)
    
    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        if not feat_list:
            raise ValueError("No features to fuse")
        # 각 모달을 sequence로 취급
        feats = torch.stack(feat_list, dim=1)  # (B, num_modals, D)
        attn_out, _ = self.mha(feats, feats, feats)
        return self.proj(attn_out.mean(dim=1))
    
class TransformerFusion(nn.Module):
    def __init__(self, feat_dims, fusion_dim, num_layers=2):
        d_model = sum(feat_dims)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        super().__init__()
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(d_model, fusion_dim)
    
    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        if not feat_list:
            raise ValueError("No features to fuse")
        x = torch.cat(feat_list, dim=1).unsqueeze(-1)  # (B, total_dim, 1)
        x = self.transformer(x.transpose(1,2)).transpose(1,2)
        return self.proj(self.pool(x).squeeze(-1))

class HypernetFusion(nn.Module):
    def __init__(self, feat_dims, fusion_dim):
        super().__init__()
        self.hypernet = nn.Sequential(
            nn.Linear(sum(feat_dims), 128),
            nn.ReLU(),
            nn.Linear(128, sum(feat_dims) * fusion_dim),
            nn.Unflatten(-1, (sum(feat_dims), fusion_dim))
        )
        self.fusion = nn.Linear(sum(feat_dims), fusion_dim)
    
    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        if not feat_list:
            raise ValueError("No features to fuse")
        concat = torch.cat(feat_list, dim=1)
        weights = self.hypernet(concat)  # 동적 가중치 생성
        weighted_feats = concat.unsqueeze(-1) * weights
        return self.fusion(weighted_feats.mean(-1).sum(1))

class ModalMoE(nn.Module):
    def __init__(self, feat_dims, fusion_dim, top_k=2):
        super().__init__()
        self.num_experts = len(feat_dims)
        self.gate = nn.Linear(sum(feat_dims), self.num_experts)  # **router**
        self.experts = nn.ModuleList([
            nn.Linear(dim, fusion_dim) for dim in feat_dims
        ])
        self.top_k = top_k
    
    def forward(self, feat_list):
        concat = torch.cat(feat_list, dim=1)
        gates = F.softmax(self.gate(concat), dim=-1)  # (B, num_experts)
        
        # Top-k gating: **몇몇 모달만 선택**
        topk_gates, topk_indices = gates.topk(self.top_k, dim=-1)
        
        output = torch.zeros(feat_list[0].shape[0], self.experts[0].out_features,
                           device=feat_list[0].device)
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]
            gate_val = topk_gates[:, i].unsqueeze(-1)
            # 선택된 expert만 실행
            for b in range(expert_idx.shape[0]):
                expert = self.experts[expert_idx[b]]
                feat = feat_list[expert_idx[b]][b:b+1]
                output[b] += gate_val[b] * expert(feat)
        return output


class SigmoidModalFusion(nn.Module):
    def __init__(self, feat_dims, fusion_dim):
        super().__init__()
        self.num_modals = len(feat_dims)
        self.gate = nn.Sequential(
            nn.Linear(sum(feat_dims), 64),
            nn.ReLU(),
            nn.Linear(64, self.num_modals),
            nn.Sigmoid()  # [0,1] 가중치
        )
        self.fusion = nn.Linear(sum(feat_dims), fusion_dim)
    
    def forward(self, feat_list):
        concat = torch.cat(feat_list, dim=1)  # (B, total_dim)
        gates = self.gate(concat)             # (B, num_modals)
        
        # 벡터화된 weighted sum (빠름!)
        weighted = sum(gates[:, i].unsqueeze(-1) * feat_list[i] 
                      for i in range(self.num_modals))
        
        return self.fusion(weighted)