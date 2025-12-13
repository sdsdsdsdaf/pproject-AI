from itertools import product
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from Utils.Loss import multitask_loss
from Utils.util import multitask_f1_macro, print_dict_structure
import numpy as np
import pandas as pd
import random
import os
import time
from Utils.LifelogDataset import DailyLifelogDataset
from Models.MultiModal import MultimodalModel
from Models.Encoder import *
from Utils.util import train, custom_collate_fn, init_weight
import pickle as pkl
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import wandb

# SEED 고정
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":  

    required_mods = ['mLight','mACStatus', 'mActivity', 'mBle', 'mGps', 'wHr',
                     'mScreenStatus', 'mUsageStats', 'mWifi', 'wLight', 'wPedo']
    feature_dims = {
        "mLight": 2,
        "mBle": 12,
        "mACStatus": 2,
        "wPedo": 14,
        "mGps": 14,
        "mUsageStats": 6,
        "mActivity": 2,
        "wHr": 8,
        "wLight": 2,
        "mWifi": 8,
        "mScreenStatus": 2,
    }

    encoder_type_table = {
        "mlp": MLPEncoder,
        "conv": SimpleEncoder1D,
        "lstm": SimpleLSTMEncoder1D,
        "lstm+attention": SimpleLSTMAttnEncoder1D,
        "transformer": SimpleTransformerEncoder1D
    }

    
    DROPOUT_RATIO = 0.1
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = {k:288 for k in required_mods}  # 5min 간격, 24*60/5 = 288
    LATENT_DIM = 32
    FUSION_DIM = 128
    BS = 32
    RANDOM_STATE = 42
    ENCODER_TYPE = "transformer" # [mlp, conv, lstm, lstm+attention, transformer]
    FUSION_TYPE = "modal_gating" # [concat, advanced_mlp, modal_gating, sigmoid_modal, cross_attention, moe]
    set_seed(RANDOM_STATE)


    EPOCH = 50

    #WandB init     
    USE_WANDB = True
    

    # ---- 1) DATA LOAD ----
    train_data_dict = pkl.load(open('ETRI 2024\processed_linear_5min_10%_MIN_MASK_dataset.pkl', 'rb'))

    required_mods = ['mLight','mACStatus', 'mActivity', 'mGps', 'wHr',
                     'mScreenStatus', 'mUsageStats', 'mWifi', 'wLight', 'wPedo']

    full_dataset = DailyLifelogDataset(
        data_dict=train_data_dict,
        csv_path="ETRI 2024\ch2025_metrics_train.csv",
        required_mods=required_mods
    )

    train_test_split_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    train_dataset = Subset(full_dataset, train_test_split_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True, collate_fn=custom_collate_fn)

    # ---- 2) MODEL INIT ----
    modality_names = required_mods

    ENCODER_TYPE_LIST = ["mlp", "conv", "lstm", "lstm+attention", "transformer"]
    FUSION_TYPE_LIST = ["concat", "advanced_mlp", "modal_gating", "sigmoid_modal", "cross_attention", "moe"]

    for i, (encoder_type, fusion_type) in enumerate(product(ENCODER_TYPE_LIST, FUSION_TYPE_LIST)):
        print(f"\n-------------------------------- [{i+1}/{len(ENCODER_TYPE_LIST)*len(FUSION_TYPE_LIST)}] --------------------------------")
        print(f"Encoder Type: {encoder_type}, Fusion Type: {fusion_type}\n")

        run_name = f"{encoder_type}_{fusion_type}"
        config = {
            "dropout": DROPOUT_RATIO,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "input_dim": INPUT_DIM,
            "latent_dim": LATENT_DIM,
            "fusion_dim": FUSION_DIM,
            "batch_size": BS,
            "epochs": EPOCH,
            "device": str(DEVICE),
            "modalities": list(required_mods),
            "encoder_type": encoder_type,
            "fusion_type": fusion_type,
            "exp_id": i+1
        }

        if USE_WANDB:
            wandb.init(
                project="ETRI-Multimodal",
                name=run_name,
                config=config,
                reinit=True
            )

            wandb.run.tags = [
                encoder_type,
                fusion_type,
                "ablation",
            ]
            
        encoder_dict = {}
        for modality_name in modality_names:
            encoder_dict[modality_name] = encoder_type_table[encoder_type](
                in_channels=feature_dims[modality_name],
                latent_dim=LATENT_DIM,
                out_dim=FUSION_DIM,
            )

        model = MultimodalModel(
            modality_names=modality_names,
            encoder_dict=encoder_dict,
            fusion_dim=FUSION_DIM,
            dropout_ratio=DROPOUT_RATIO,
            fusion_type=fusion_type,
            fusion_layer_kwargs={},
            task_class_dict={"Q1":2, "Q2":2, "Q3":2, "S1":3, "S2":2, "S3":2}
        ).to(DEVICE)

        # ---- 3) TRAINING ----
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = multitask_loss()
        result = train(
            model=model,
            train_loader=train_loader,
            test_loader=val_loader,
            optimizer=optimizer,
            device=DEVICE,
            epoch=EPOCH,
            criterion=loss_fn,
            metric="f1_macro",
            weight_dir="Weights/ETRI2024",
            use_wandb=USE_WANDB
        )

        if USE_WANDB:
            wandb.summary["best_metric"] = result["best_metric"]
            wandb.summary["best_val_loss"] = result["best_val_loss"]
            wandb.summary["encoder"] = encoder_type
            wandb.summary["fusion"] = fusion_type
            wandb.finish()