import torch
import torch.nn as nn
import torch.nn.functional as F

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

class BasicMultimodalModel(MultiModalAbstractModel):  # 오타 수정 가정
    def __init__(self, 
                 modality_names: list[str], 
                 encoder_dict: dict[str, nn.Module],
                 fusion_dim: int = 128,
                 dropout_ratio: float = 0.1,
                 task_class_dict: dict[str, int] = {"Q1":2, "Q2":2, "Q3":2, "S1":3, "S2":2, "S3":2},
                 act_fn=nn.GELU):
        super().__init__()
        
        # 총 차원 계산
        total_dim = sum(enc.out_dim for enc in encoder_dict.values())
        
        # 모듈 조립
        self.encoder_handler = ModalityEncoderHandler(modality_names, encoder_dict)
        self.fusion = ConcatFusion(total_dim, fusion_dim, act_fn)
        self.classifier = MultiHeadClassifier(fusion_dim, task_class_dict, dropout_ratio, act_fn)
    
    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feats = self.encoder_handler(data)
        fused = self.fusion(feats)
        
        return self.classifier(fused)


