import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicMultimodalModel(nn.Module):
    def __init__(self, modality_names, encoder_dict, act_fn=nn.GELU,fusion_dim=128, dropout_ratio=0.1, task_clsss_dict:dict[str, int]={"Q1":2, "Q2":2, "Q3":2, "S1":3, "S2":2, "S3":2}):
        super().__init__() 
        
        self.modality_names = modality_names
        self.encoders = nn.ModuleDict(encoder_dict)

        total_dim = sum(enc.out_dim for enc in encoder_dict.values())
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            act_fn(),
        )

        self.multihead_classifiers = nn.ModuleDict()
        for q, num_classes in task_clsss_dict.items():
            self.multihead_classifiers[q] = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim //2),
                nn.Dropout(dropout_ratio) if dropout_ratio > 0 else nn.Identity(),
                act_fn(),
                nn.Linear(fusion_dim //2, num_classes if num_classes > 2 else 1),
            )

    def forward(self, data:dict[str, torch.Tensor]):
        x_list = []

        for m in self.modality_names:
            if m not in data.keys():
                continue
            x = data[m]       # (B, C, T)
            if x.ndim == 2:
                x = x.unsqueeze(0) # (C, T) -> (1, C, T)

            enc = self.encoders[m](x)
            x_list.append(enc)

        fused = torch.cat(x_list, dim=1)
        h = self.fusion(fused)
        
        out = {}
        for q, classifier in self.multihead_classifiers.items():
            out[q] = classifier(h)

        return out

    @torch.no_grad()
    def predict(self, data:dict[str, torch.Tensor]):

        """
        Predict the output of the model for each task
        Batch size of the data is 1

        Args:
            data: dict[str, torch.Tensor]
            - key: modality name
            - value: tensor of shape (1, C, T)
        Returns:
            dict[str, int]
            - key: task name
            - value: predicted class
        """
        
        self.eval()
        output:dict[str, torch.Tensor] = self.forward(data)
        for k, v in output.items():
            if k == "S1":
                output[k] = output[k].argmax().int().item()
            else:
                output[k] = (output[k]>0.5).int().item()
        return output
