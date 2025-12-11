import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
import datetime as dt
from tqdm import tqdm
import re


class DailyLifelogDataset(Dataset):
    def __init__(self, data_dict, csv_path, required_mods=None):
        """
        data_dict structure:
        {
           (subject_id, date): {
               modality: (df, mask_df),
               ...
           }
        }
        csv_path: 'ch2025_metrics_train.csv'
        """
        self.data_dict = data_dict
        self.required_mods = set(required_mods) if required_mods else None

        # ---- 1) CSV Load and make label lookup ----
        df = pd.read_csv(csv_path)
        df["lifelog_date"] = pd.to_datetime(df["lifelog_date"]).dt.date
        df["subject_id"] = df["subject_id"].astype(str)

        self.label_map = {}
        for _, row in df.iterrows():
            key = (row["subject_id"], row["lifelog_date"].isoformat())
            self.label_map[key] = torch.tensor(
                [row["Q1"], row["Q2"], row["Q3"], row["S1"], row["S2"], row["S3"]],
                dtype=torch.long
            )

        print("[Dataset] Precomputing tensors (this runs once)...")

        self.tensor_dict = {}   # ← 최종적으로 Dataset은 이것만 사용

        for key, modalities in tqdm(data_dict.items(), leave=False):
            subj, date = key

            # label 없는 샘플 제외
            if (subj, date) not in self.label_map:
                continue

            # required_mods 전체 포함 여부 검사
            if self.required_mods:
                mods_in_sample = set(modalities.keys())
                if not self.required_mods.issubset(mods_in_sample):
                    continue

            # 이 key는 사용할 수 있음 → 텐서로 변환 시작
            self.tensor_dict[key] = {}

            for mod in self.required_mods:
                df, mask_df = modalities[mod]  # (DataFrame, DataFrame)

                # 여기서 단 1회만 tensor 변환 (getitem에서 반복하지 않음)
                data_tensor = torch.tensor(df.values, dtype=torch.float32).T
                mask_tensor = torch.tensor(mask_df.values, dtype=torch.float32).T

                # 모달리티별 완성 텐서 저장
                self.tensor_dict[key][mod] = torch.cat([data_tensor, mask_tensor], dim=0)

        # ----------------------------------------------
        # 3) valid_keys 목록 확정
        # ----------------------------------------------
        self.valid_keys = list(self.tensor_dict.keys())

        print(f"[Dataset] Valid samples = {len(self.valid_keys)} / {len(data_dict)}")
        print("[Dataset] Precompute finished.\n\n")

    def __len__(self):
        return len(self.valid_keys)

    def __getitem__(self, idx):
        key = self.valid_keys[idx]

        sample = self.tensor_dict[key]

        return {
            "data": sample,                # ← 이미 tensor
            "label": self.label_map[key],  # ← 이미 tensor
            "key": key
        }

class GANImputateDataset(Dataset):

    """
    Output Data Shape
    Dict[
        (subject_id, date):{
            '(modality)':
                '(data, mask)'
        }
    ]
    """

    def __init__(self, data_dir, frequency="5min"):
        files = os.listdir(data_dir)
        self.data = []

        df = {}
        self.frequency = frequency
        for file in files:
            file_path = os.path.join(data_dir, file)
            modality = re.search(r'ch2025_([^_]+)_zero', file_path).group(1)
            df[modality] = pd.read_csv(file_path)
            df[modality]['timestamp'] = pd.to_datetime(df[modality]['timestamp'])





if __name__ == "__main__":
    import pickle as pkl
    data_dict = pkl.load(open("ETRI 2024\processed_linear_5min_10%_MIN_MASK_dataset.pkl", "rb"))
    required_mods = [
        'mLight','mACStatus', 'mActivity', 'mBle', 'mGps', 'wHr',
                  'mScreenStatus', 'mUsageStats', 'mWifi', 'wLight', 'wPedo'
    ]
    ds = DailyLifelogDataset(data_dict, "ETRI 2024\ch2025_metrics_train.csv", required_mods=required_mods)
    print(f"Dataset size: {len(ds)}")
    sample, label, key = ds[0]["data"], ds[0]["label"], ds[0]["key"]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Label: {label}, Key: {key}")
    for mod, tensor in sample.items():
        print(f"Modality: {mod}, Tensor shape: {tensor.shape}")

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    for batch in loader:
        batch_data = batch["data"]
        print(f"Batch data keys: {list(batch_data.keys())}")