import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np

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
            key = (row["subject_id"], row["lifelog_date"])
            self.label_map[key] = torch.tensor(
                [row["Q1"], row["Q2"], row["Q3"], row["S1"], row["S2"], row["S3"]],
                dtype=torch.long
            )

        # ---- 2) Filter samples that have all required modalities + also have label ----
        self.valid_keys = []
        for key, modalities in data_dict.items():
            subj, date = key

            # label 존재?
            if (subj, date) not in self.label_map:
                continue

            # modality 존재?
            if self.required_mods:
                mods_in_sample = set(modalities.keys())
                if not self.required_mods.issubset(mods_in_sample):
                    continue

            self.valid_keys.append(key)

        print(f"[Dataset] Valid samples = {len(self.valid_keys)} / {len(data_dict)}")

    def __len__(self):
        return len(self.valid_keys)

    def __getitem__(self, idx):
        key = self.valid_keys[idx]
        modality_dict = self.data_dict[key]

        sample = {}

        # ---- 3) modality → tensor(C+Cmask, T) ----
        for mod in self.required_mods:
            df, mask_df = modality_dict[mod]

            data_tensor = torch.tensor(df.values, dtype=torch.float32).T
            mask_tensor = torch.tensor(mask_df.values, dtype=torch.float32).T

            sample[mod] = torch.cat([data_tensor, mask_tensor], dim=0)

        # ---- 4) label 가져오기 ----
        label = self.label_map[key]

        return sample, label, key


if __name__ == "__main__":
    import pickle as pkl
    data_dict = pkl.load(open("ETRI 2024\processed_linear_5min_dataset.pkl", "rb"))
    required_mods = [
        'mLight','mACStatus', 'mActivity', 'mBle', 'mGps', 'wHr',
                  'mScreenStatus', 'mUsageStats', 'mWifi', 'wLight', 'wPedo'
    ]
    ds = DailyLifelogDataset(data_dict, "ETRI 2024\ch2025_metrics_train.csv", required_mods=required_mods)
    print(f"Dataset size: {len(ds)}")
    sample, label, key = ds[0]
    print(f"Sample keys: {list(sample.keys())}")
    print(f"Label: {label}, Key: {key}")
    for mod, tensor in sample.items():
        print(f"Modality: {mod}, Tensor shape: {tensor.shape}")