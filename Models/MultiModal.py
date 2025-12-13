import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from Models.FusionLayer import FusionFactory, ConcatFusion
except ImportError:
    from FusionLayer import FusionFactory, ConcatFusion
try:
    from Models.Encoder import EncoderAbstractModel
except ImportError:
    from Encoder import EncoderAbstractModel


class MultiModalAbstractModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data:dict[str, torch.Tensor]):
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, data:dict[str, torch.Tensor]):

        """
        Predict the output of the model for each task
        Batch size of the data is 1

        Args:
            data (dict[ key [str]: modality namevalue: tensor of shape [1, C, T]]): input_data
                
        Returns:
            dict[str, int]
            key: task name
            value: predicted class
        """
        
        self.eval()
        for k, v in data.items():
            if v.ndim == 2:
                data[k] = v.unsqueeze(0)

        
        preds:dict[str, torch.Tensor] = self.forward(data)
        outputs:dict[str, int]= {}
        for k, v in preds.items():
            if k == "S1":
                outputs[k] = preds[k].argmax().int().item()
            else:
                outputs[k] = (preds[k]>0.5).int().item()
        return outputs

class ModalityEncoderHandler(nn.Module):
    def __init__(self, modality_names: list[str], encoders: dict[str, nn.Module]):
        super().__init__()
        self.modality_names = modality_names
        self.encoders = nn.ModuleDict(encoders)
    
    def forward(self, data: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        feats = []
        for modality in self.modality_names:
            if modality not in data or data[modality].numel() == 0:
                continue
            x = data[modality]
            if x.ndim == 2:
                x = x.unsqueeze(0)  # (C, T) -> (1, C, T)
            elif x.ndim == 1:
                x = x.unsqueeze(0).unsqueeze(0)  # (T,) -> (1, 1, T)
            feats.append(self.encoders[modality](x))
        return feats


# 3. Multi-head Classifier
class MultiHeadClassifier(nn.Module):
    def __init__(self, fusion_dim: int, task_class_dict: dict[str, int], 
                 dropout_ratio: float = 0.1, act_fn=nn.GELU):
        super().__init__()
        self.heads = nn.ModuleDict()
        for task, num_classes in task_class_dict.items():
            head = [
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
                act_fn(),
                nn.Linear(fusion_dim // 2, num_classes if num_classes > 2 else 1),
            ]
            self.heads[task] = nn.Sequential(*head)
    
    def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        return {task: head(fused) for task, head in self.heads.items()}

class MultimodalModel(MultiModalAbstractModel):  # 오타 수정 가정
    def __init__(self, 
                 modality_names: list[str], 
                 encoder_dict: dict[str, EncoderAbstractModel],
                 fusion_dim: int = 128,
                 dropout_ratio: float = 0.1,
                 task_class_dict: dict[str, int] = {"Q1":2, "Q2":2, "Q3":2, "S1":3, "S2":2, "S3":2},
                 act_fn=nn.GELU,
                 fusion_type="concat",
                 fusion_layer_kwargs={}):
        
        """
        Late Fusion 기반 Multi-task 멀티모달 모델
        
        1. 각 모달 → 독립 Encoder → 특징 추출
        2. 여러 Fusion 전략으로 특징 융합  
        3. Multi-head classifier로 병렬 task 예측
        
        Args:
            modality_names (list[str]): 사용 모달리티 목록 ['image', 'text', 'audio']
            encoder_dict (dict[str, EncoderAbstractModel]): 
                {모달이름: Encoder} 딕셔너리 (각 Encoder는 out_dim 속성 필수)
            fusion_dim (int): Fusion 출력 차원 (기본 128)
            dropout_ratio (float): Classifier dropout (기본 0.1)
            task_class_dict (dict[str, int]): 
                {task명: 클래스수} ex) {"Q1":2, "S1":3} (2=이진, 3=3분류)
            act_fn: 활성화 함수 (기본 GELU)

            fusion_type (str): Fusion 종류
                - "concat": 단순 연결 MLP [Group 1]
                - "advanced_mlp": 깊은 MLP + LayerNorm [Group 1] 
                - "modal_gating": Softmax 모달 가중합 [Group 2]
                - "sigmoid_modal": Sigmoid 모달 가중합 [Group 2]
                - "cross_attention": 모달 간 Self-Attention [Group 3]
                - "hypernet": 동적 가중치 생성 Hypernetwork [Group 2]
                - "moe": Top-k Mixture of Experts [Group 2]
            
            
            **kwargs: 추가 하이퍼파라미터
                - advanced_mlp: dropout=0.1, act_fn=nn.GELU
                - cross_attention: num_heads=8
                - moe: top_k=2
        
        Example:
            >>> encoders = {'img': ConvEncoder(out_dim=64), 'txt': LSTMEncoder(out_dim=64)}
            >>> model = BasicMultimodalModel(['img','txt'], encoders, fusion_type="sigmoid_modal")
            >>> outputs = model({'img': img_data, 'txt': txt_data})
            >>> preds = model.predict({'img': img_data, 'txt': txt_data})  # dict[str,int]
        """

        super().__init__()
        
        # 총 차원 계산
        total_dim = sum(enc.out_dim for enc in encoder_dict.values())
        
        # 모듈 조립
        self.encoder_handler = ModalityEncoderHandler(modality_names, encoder_dict)
        self.fusion = FusionFactory.create(
            fusion_type=fusion_type,
            feat_dims=[enc.out_dim for enc in encoder_dict.values()],
            fusion_dim=fusion_dim,
            total_dim=total_dim,
            act_fn=act_fn,
            **fusion_layer_kwargs
        )

        self.classifier = MultiHeadClassifier(fusion_dim, task_class_dict, dropout_ratio, act_fn)
    
    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feats = self.encoder_handler(data)
        fused = self.fusion(feats)
        
        return self.classifier(fused)


