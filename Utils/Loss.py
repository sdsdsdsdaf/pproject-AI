import torch
import torch.nn as nn
import torch.nn.functional as F

class multitask_loss(nn.Module):
    
    def __init__(
        self,
        criterion_dict={
            "Q1":F.binary_cross_entropy_with_logits, 
            "Q2":F.binary_cross_entropy_with_logits, 
            "Q3":F.binary_cross_entropy_with_logits, 
            "S1":F.cross_entropy, 
            "S2":F.binary_cross_entropy_with_logits, 
            "S3":F.binary_cross_entropy_with_logits
        },
        reduction="mean"
    ):
        
        super().__init__()
        self.criterion_dict = criterion_dict
        self.reduction = reduction
        self.str_to_num = {
            "Q1": 0,
            "Q2": 1,
            "Q3": 2,

            "S1": 3,
            "S2": 4,
            "S3": 5
        }
    
    def forward(self, outputs:dict[str, torch.Tensor], labels:torch.Tensor):
        criterion_dict = self.criterion_dict
        reduction = self.reduction
        task_losses = {task: 0.0 for task in outputs}

        loss = 0.0
        for task in outputs:
            y_true:torch.Tensor = labels[:, self.str_to_num[task]].squeeze()
            if task == "S1":      # multi-class classification
                y_true = y_true.long()
            else:                 # binary tasks (BCE)
                y_true = y_true.float()
            task_losses[task] = criterion_dict[task](outputs[task].squeeze(), y_true)
            loss += task_losses[task]
        
        if reduction == "mean":
            loss = loss / len(outputs)

        return loss, task_losses