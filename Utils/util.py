import ast
from collections import defaultdict
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

try:
    from Utils.Loss import multitask_loss
except ImportError:
    from Loss import multitask_loss

from sklearn.metrics import f1_score


def custom_collate_fn(batch):
    keys = [item["key"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])

    data_batch = {
        mod: torch.stack([item["data"][mod] for item in batch])
        for mod in batch[0]["data"]
    }

    return {"data": data_batch, "label": labels, "key": keys}

def multitask_f1_macro(outputs, labels, task_classes):
    """
    outputs : dict(task → logits tensor(B,num_classes))
    labels  : dict(task → gt tensor(B))
    task_classes: dict(task → num_classes)
    """
    f1_list = {}

    for task, num_classes in task_classes.items():
       
        label = labels[task].detach().cpu().numpy()
        output = outputs[task].detach().cpu().numpy()

        # ----- 2) f1 calculation -----
        if num_classes == 2:
            f1 = f1_score(label, output, average="binary")
        else:
            f1 = f1_score(label, output, average="macro")

        f1_list[task] = f1

    # ----- 3) macro over tasks -----
    macro_f1 = sum(f1_list.values()) / len(f1_list)

    return macro_f1, f1_list


def print_dict_structure(d, indent=0, max_value_length=None):
    """Print dictionary structure without expanding large values."""
    prefix = "  " * indent

    if not isinstance(d, dict):
        print(f"{prefix}- {type(d).__name__}")
        return

    for i, (key, value) in enumerate(d.items()):
        if max_value_length and i >= max_value_length:
            print(f"{prefix}... (and more keys not shown)")
            break
        key_str = str(key)
        if len(key_str) > 60:  # 너무 긴 key는 잘라줌
            key_str = key_str[:60] + "..."

        # value 타입만 출력
        if isinstance(value, dict):
            print(f"{prefix}{key_str}/   (dict)")
            print_dict_structure(value, indent + 1)
        else:
            print(f"{prefix}{key_str}: {type(value).__name__}")


def safe_to_list(x):
    """
    문자열 → list 변환
    numpy array → list 변환
    이미 list / tuple ⇒ 그대로
    그 외 모든 타입 ⇒ 빈 list 반환
    """

    # NaN, None 처리
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []

    # 문자열 처리
    if isinstance(x, str):
        try:
            obj = ast.literal_eval(x)
            # dict 한 개가 들어있다 → 리스트로 감싸기
            if isinstance(obj, dict):
                return [obj]
            # numpy array → list
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # list면 그대로
            if isinstance(obj, list):
                return obj
            return []
        except Exception:
            return []

    # numpy array
    if isinstance(x, np.ndarray):
        return x.tolist()

    # 목록 타입
    if isinstance(x, (list, tuple)):
        return list(x)

    # dict 단일 객체
    if isinstance(x, dict):
        return [x]

    # 나머지 타입 → 처리 불가
    return []

def evaluate_model(model:nn.Module, data_loader, criterion, device, task_classes, metric="f1_macro", use_wandb=False):
    model.eval()
    total_loss = 0

    # ✨ task별 y_pred / y_true 저장용
    preds = defaultdict(list)
    trues = defaultdict(list)

    def str_to_num(task:str):
        mapping = {
            "Q1": 0,
            "Q2": 1,
            "Q3": 2,
            "S1": 3,
            "S2": 4,
            "S3": 5
        }
        return mapping[task]

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
             # --- Move data to device ---
            for m in batch["data"]:
                batch["data"][m] = batch["data"][m].to(device)

            batch['label'] = batch['label'].to(device)

            outputs = model(batch["data"])
            loss, _ = criterion(outputs, batch["label"])
            total_loss += loss.item()

            # 🔥 각 task별 예측값/정답 저장
            for task, out in outputs.items():
                out = out.squeeze()

                # BCE → sigmoid로 확률 변환 + threshold
                if task != "S1":  
                    prob = torch.sigmoid(out)
                    pred = (prob > 0.5).long().cpu()

                # CE → argmax
                else:
                    pred = torch.argmax(out, dim=-1).cpu()

                true = batch["label"][:, str_to_num(task)].cpu()

                # CE는 true가 long이어야 하고 BCE는 float → long으로 통일
                trues[task].append(true.long())
                preds[task].append(pred)

    # 텐서 concat
    preds = {t: torch.cat(preds[t]) for t in preds}
    trues = {t: torch.cat(trues[t]) for t in trues}
    task_dict = {
        "Q1":2,
        "Q2":2,
        "Q3":2,
        "S1":3,
        "S2":2,
        "S3":2
    }
    # F1 계산
    if metric == "f1_macro":
        macro_f1, f1_each = multitask_f1_macro(preds, trues, task_dict)
    elif metric == "f1_each":
        for t in task_classes:
            f1_each[t] = (preds[t].argmax(dim=1) == trues[t]).float().mean().item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


    return {
        "loss": total_loss / preds[list(preds.keys())[0]].shape[0],
        "f1_macro": macro_f1,
        "f1_each": f1_each
    }



def train_one_epoch(model:nn.Module, train_loader:DataLoader, criterion, optimizer:torch.optim.Optimizer, device:str|torch.device, use_wandb=False):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        # --- Move data to device ---
        for m in batch["data"]:
            batch["data"][m] = batch["data"][m].to(device)

        batch['label'] = batch['label'].to(device)

        # Forward
        outputs = model(batch['data'])  # <-- 중요! now correct
        loss, task_losses = criterion(outputs, batch['label'])
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)

        if use_wandb:
            # 🔥 step 단위 로깅 (task별 loss)
            wandb.log({f"task_loss/{k}": v for k, v in task_losses.items()})

            # 🔥 learning rate 로깅 (batch 단위)
            wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]})
            wandb.log({"grad_norm": total_norm})

    return total_loss / len(train_loader)



def train(model:nn.Module, epoch:int, train_loader:DataLoader, test_loader:DataLoader, criterion, optimizer:torch.optim.Optimizer, device:str|torch.device, metric="f1_macro", weight_dir="ETRI 2024/Weights", use_wandb=False):
    model.train()
    train_loss_list = []
    test_loss_list = []
    test_metric_list = []
    best_metric = 0.0
    os.makedirs(weight_dir, exist_ok=True)
    file_path = os.path.join(weight_dir, "best_model.pth")
    for E in range(1, epoch+1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, use_wandb=use_wandb)
        eval_result = evaluate_model(model, test_loader, criterion, device, metric=metric, task_classes=[2, 2, 2, 3, 2, 2], use_wandb=use_wandb)
        print(f"Epoch [{E}/{epoch}] - Train Loss: {train_loss:.4f}, Test Loss: {eval_result['loss']:.4f}, Test {metric}: {eval_result[metric]:.4f}")

        if eval_result[metric] > best_metric:

            best_metric = eval_result[metric]
            torch.save(model.state_dict(), file_path)
            print(f"  -> New best model saved with {metric}: {best_metric:.4f}")

        train_loss_list.append(train_loss)
        test_loss_list.append(eval_result['loss'])
        test_metric_list.append(eval_result[metric])

        # 🔥 epoch 단위 로깅
        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": eval_result['loss'],
                f"val_{metric}": eval_result[metric],
            })
        
    return best_metric, train_loss_list, test_loss_list, test_metric_list
