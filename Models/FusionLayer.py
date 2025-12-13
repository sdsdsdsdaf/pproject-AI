import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionFactory:
    @staticmethod
    def create(fusion_type: str, feat_dims: list[int], fusion_dim: int, 
               total_dim: int = None, **kwargs):
        
        """
        멀티모달 Fusion 모듈 팩토리 함수
        
        모든 Fusion 모듈을 통일된 인터페이스로 생성합니다.
        
        Args:
            fusion_type (str): Fusion 종류
                - "concat": 단순 연결 MLP [Group 1]
                - "advanced_mlp": 깊은 MLP + LayerNorm [Group 1] 
                - "modal_gating": Softmax 모달 가중합 [Group 2]
                - "sigmoid_modal": Sigmoid 모달 가중합 [Group 2]
                - "cross_attention": 모달 간 Self-Attention [Group 3]
                - "hypernet": 동적 가중치 생성 Hypernetwork [Group 2]
                - "moe": Top-k Mixture of Experts [Group 2]
            
            feat_dims (list[int]): 각 모달리티의 출력 차원 [enc.out_dim for enc in encoders]
                ex) [64, 32, 16] = 이미지64차원, 텍스트32차원, 오디오16차원
            
            fusion_dim (int): 최종 Fusion 출력 차원 (보통 128)
            
            total_dim (int, optional): sum(feat_dims)의 shortcut. 자동 계산됨
            
            **kwargs: 추가 하이퍼파라미터
                - advanced_mlp: dropout=0.1, act_fn=nn.GELU
                - cross_attention: num_heads=8
                - moe: top_k=2
        
        Returns:
            nn.Module: feat_list[list[torch.Tensor]] -> (B, fusion_dim)
            
        Raises:
            ValueError: 알 수 없는 fusion_type
        
        Examples:
            >>> feat_dims = [64, 32, 16]  # 3개 모달
            >>> fusion = FusionFactory.create("sigmoid_modal", feat_dims, 128)
            >>> feats = [img_feat, txt_feat, audio_feat]  # list of (B, Di)
            >>> fused = fusion(feats)  # (B, 128)
            
            >>> # 실험하며 쉽게 교체
            >>> fusion = FusionFactory.create("cross_attention", feat_dims, 128, num_heads=4)
        
        Notes:
            - Group 1 (total_dim): concat 기반 MLP 계열
            - Group 2 (feat_dims): 모달별 처리 인식  
            - Group 3 (feat_dims+total_dim): Attention 계열
        """
        
        total_dim = total_dim or sum(feat_dims)
        feat_dims = feat_dims or [total_dim]  # backward compat
        fusion_type = fusion_type.lower()
        if fusion_type not in ["concat", "advanced_mlp", "modal_gating", "sigmoid_modal",
                               "cross_attention", "hypernet", "moe"]:
            raise ValueError(f"Unknown fusion type: {fusion_type}")


        if fusion_type == "concat":
            return ConcatFusion(total_dim, fusion_dim)
        elif fusion_type == "advanced_mlp":
            return AdvancedMLPFusion(total_dim, fusion_dim, **kwargs)
        elif fusion_type == "modal_gating":
            return ModalGatingFusion(feat_dims, fusion_dim)
        elif fusion_type == "sigmoid_modal":
            return SigmoidModalFusion(feat_dims, fusion_dim)
        elif fusion_type == "cross_attention":
            return CrossModalAttention(feat_dims, fusion_dim, total_dim, **kwargs)
        elif fusion_type == "hypernet":
            return HypernetFusion(feat_dims, fusion_dim)
        elif fusion_type == "moe":
            return ModalMoE(feat_dims, fusion_dim, **kwargs)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")




# 2. Fusion 모듈
class ConcatFusion(nn.Module):
    def __init__(self, total_dim: int, fusion_dim: int, act_fn=nn.GELU):

        """
        기본 Late Fusion: 모든 모달 특징을 단순 연결(concat) 후 MLP

        Args:
            total_dim (int): 각 모달리티의 임베딩 차원을 전부 합친 값
            fusion_dim (int): 최종 차원
            act_fn (nn.Module): 활성화 함수
    
        """
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
    
    def __init__(self, total_dim, fusion_dim, dropout=0.1, act_fn=nn.GELU):
        
        """
        Concat Fusion 개선판: 깊은 MLP + LayerNorm + Dropout (2->3층)
        과적합 방지 + 표현력 강화

        Args:
            total_dim (int): 각 모달리티의 임베딩 차원을 전부 합친 값
            fusion_dim (int): 최종 차원
            dropout (float): Dropout 비율
            act_fn (nn.Module): 활성화 함수
        """

        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            act_fn(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            act_fn(),
        )
    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        if not feat_list:
            raise ValueError("No features to fuse")
        return self.fusion(torch.cat(feat_list, dim=1))


class ModalGatingFusion(nn.Module):
    
    def __init__(self, feat_dims:list[int], fusion_dim:int):

        """
        Softmax 기반 모달 가중합: 각 모달에 중요도 할당 (sum_i=1)

        Args:
            feat_dims (list[int]): 각 모달의 차원
            fusion_dim (int): 최종 차원
        """

        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(sum(feat_dims), len(feat_dims)),
            nn.Softmax(dim=-1)
        )
        max_dim = max(feat_dims)
        self.fusion = nn.Linear(max_dim, fusion_dim)
    
    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        if not feat_list:
            raise ValueError("No features to fuse")
        concat = torch.cat(feat_list, dim=1)
        weights = self.gate(concat)  # (B, num_modals)
        weighted = sum(weights[:, i].unsqueeze(-1) * feat_list[i] for i in range(len(feat_list)))
        return self.fusion(weighted)


class CrossModalAttention(nn.Module):
    
    def __init__(self, feat_dims:list[int], fusion_dim:int, total_dim:int|None=None, num_heads=8, act_fn=nn.GELU):

        """
        모달 간 Self-Attention: 각 모달이 서로를 참고 (Transformer 스타일)

        Args:
            feat_dims (list[int]): 각 모달의 차원
            fusion_dim (int): 최종 차원
            total_dim (int): 각 모달리티의 임베딩 차원을 전부 합친 값
            num_heads (int): Multi-head Attention의 헤드 수
        """

        super().__init__()
        self.mha = None
        self.feat_dims = feat_dims
        self.fusion_dim = fusion_dim
        self.num_heads = num_heads
        self.total_dim = total_dim
        
        if total_dim is not None:
            self.mha = nn.MultiheadAttention(
                embed_dim=total_dim//len(feat_dims),  # per modality dim
                num_heads=num_heads,
                batch_first=True
            )
        self.proj = nn.Linear(total_dim//len(feat_dims), fusion_dim)
    
    def forward(self, feat_list: list[torch.Tensor]) -> torch.Tensor:
        if not feat_list:
            raise ValueError("No features to fuse")
        if self.mha is None:
            self.mha = nn.MultiheadAttention(
                embed_dim=self.total_dim//len(self.feat_dims),  # per modality dim
                num_heads=self.num_heads,
                batch_first=True
            )

        # 각 모달을 sequence로 취급
        feats = torch.stack(feat_list, dim=1)  # (B, num_modals, D)
        (attn_out, _) = self.mha(feats, feats, feats)
        return self.proj(attn_out.mean(dim=1))
    

class HypernetFusion(nn.Module):
    def __init__(self, feat_dims, fusion_dim):

        """
        Args: 
            feat_dims (list[int]): 각 모달의 차원
            fusion_dim (int): 최종 차원
        """

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

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalMoE(nn.Module):
    def __init__(
        self,
        feat_dims,
        fusion_dim,
        num_experts=4,
        top_k=2,
        act_fn=nn.GELU,
    ):
        """
        Proper MoE with shared input & top-k routing

        Args:
            feat_dims (list[int]): modality feature dims
            fusion_dim (int): hidden dim
            num_experts (int): number of experts
            top_k (int): top-k routing
        """
        super().__init__()

        self.top_k = top_k
        self.num_experts = num_experts
        
        # modality fusion
        self.shared_proj = nn.Sequential(
            nn.Linear(sum(feat_dims), fusion_dim),
            act_fn(),
        )

        #  gate (router)
        self.gate = nn.Linear(fusion_dim, num_experts)

        #  experts (same input dim!)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                act_fn(),
            )
            for _ in range(num_experts)
        ])

    def forward(self, feat_list):
        """
        Args:
            feat_list: list of modality tensors [(B, d_i)]
        Returns:
            (B, fusion_dim)
        """
        B = feat_list[0].size(0)

        # 🔹 shared input
        x = torch.cat(feat_list, dim=1)
        h = self.shared_proj(x)  # (B, D)

        # 🔹 gating
        gate_logits = self.gate(h)          # (B, E)
        gate_probs = F.softmax(gate_logits, dim=-1)

        topk_probs, topk_idx = gate_probs.topk(self.top_k, dim=-1)  # (B, k)

        # 🔹 MoE output
        out = torch.zeros_like(h)

        # expert-wise batch routing (vectorized)
        for e in range(self.num_experts):
            mask = (topk_idx == e)          # (B, k)
            if not mask.any():
                continue

            weights = (topk_probs * mask).sum(dim=1)  # (B,)
            selected = weights > 0

            out[selected] += (
                weights[selected].unsqueeze(1)
                * self.experts[e](h[selected])
            )

        return out



class SigmoidModalFusion(nn.Module):
    def __init__(self, feat_dims, fusion_dim):

        """
        Sigmoid확률로 가중합

        Args:
            feat_dims (list[int]): 각 모달의 차원
            fusion_dim (int): 최종 차원
        """

        super().__init__()
        self.num_modals = len(feat_dims)
        self.gate = nn.Sequential(
            nn.Linear(sum(feat_dims), 64),
            nn.ReLU(),
            nn.Linear(64, self.num_modals),
            nn.Sigmoid()  # [0,1] 가중치
        )
        max_dim = max(feat_dims)
        self.fusion = nn.Linear(max_dim, fusion_dim)
    
    def forward(self, feat_list):
        concat = torch.cat(feat_list, dim=1)  # (B, total_dim)
        gates = self.gate(concat)             # (B, num_modals)
        
        weighted = sum(gates[:, i].unsqueeze(-1) * feat_list[i] 
                      for i in range(self.num_modals))
        
        return self.fusion(weighted)
    
if __name__ == "__main__":
    fusion = ModalMoE()