# %%
import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
from time import time
import datetime
import ast
import random
from Utils.preprocessing import preprocess_all_days
from tqdm.auto import tqdm
from Utils.preprocessing import preprocess_wHr, preprocess_mGps, preprocess_mWifi, preprocess_mUsage, preprocess_mBle
from Utils.util import print_dict_structure

# %% [markdown]
# # PARAMETER

# %%
FREQUENCY = 5  # minutes
INTERPOLATION = "linear"  # 'time' or 'linear'
DATA = 0
MASK = 1
MODAL_NAME = ['mLight','mACStatus', 'mActivity', 'mBle', 'mGps', 'wHr',
                  'mScreenStatus', 'mUsageStats', 'mWifi', 'wLight', 'wPedo']
MIN_RATIO = 0.1  # 최소 데이터 커버리지 비율

orginal_data_freq = {
    "mACStatus": 1,
    "mActivity": 5,
    "mBle": 10,
    "mGps": 1,
    "mLight": 10,
    "mScreenStatus": 1,
    "mUsageStats": 10,
    "mWifi": 10,
    "wHr": 1,
    "wLight": 1,
    "wPedo": 1,
}

# %% [markdown]
# # Data items
# * mACStatus
# * mActivity
# * mBle
# * mGps
# * mLight
# * mScreenStatus
# * mUsageStatus
# * wHr
# * wLight
# * wPedo
# 

# %%
SD = 42
random.seed(SD)
np.random.seed(SD)
os.environ['PYTHONHASHSEED'] = str(SD)

# %%
dataset_path = os.path.join("ETRI 2024","ch2025_data_items")
train_data_path = os.path.join("ETRI 2024","ch2025_metrics_train.csv")

# %%
print("challenge 2025 dataset " + "="*5)
for file_name in os.listdir(dataset_path):
    if file_name.endswith(".parquet"):
        print(file_name)
        
parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
print(f"\nTotal parquet files: {len(parquet_files)}")




# %% [markdown]
# # .paruet 파일 로드

# %%
# 파일 이름을 키로, DataFrame을 값으로 저장할 딕셔너리
lifelog_data = {}

# 파일별로 읽기
for file_path in parquet_files:
    name = os.path.basename(file_path).replace('.parquet', '').replace('ch2025_', '')
    lifelog_data[name] = pd.read_parquet(file_path)
    print(f"✅ Loaded: {name}, shape = {lifelog_data[name].shape}")

# %%
# 딕셔너리에 있는 모든 항목을 독립적인 변수로 할당
for key, df in lifelog_data.items():
    globals()[f"{key}_df"] = df

# %%
metric_train_df = pd.read_csv(train_data_path)
print(f"✅ Loaded: metric_train_df, shape = {metric_train_df.shape}")
print(metric_train_df.head())

# %%
sample_submission = pd.read_csv(os.path.join("ETRI 2024","ch2025_submission_sample.csv"))
sample_submission['lifelog_date'] = pd.to_datetime(sample_submission['lifelog_date'])
test_keys = set(zip(sample_submission['subject_id'], sample_submission['lifelog_date'].dt.date))
print(f"✅ Loaded: sample_submission, shape = {sample_submission.shape}")

# %%
# ✅ 분리 함수
def split_test_train(df, subject_col='subject_id', timestamp_col='timestamp'):
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df = df.dropna(subset=[timestamp_col])
    df['date_only'] = df[timestamp_col].dt.date
    df['key'] = list(zip(df[subject_col], df['date_only']))

    test_df = df[df['key'].isin(test_keys)].drop(columns=['date_only', 'key'])
    train_df = df[~df['key'].isin(test_keys)].drop(columns=['date_only', 'key'])
    return test_df, train_df


# %%
# ✅ DataFrame 별 timestamp 컬럼 수동 지정
dataframes = {
    'mACStatus': (mACStatus_df, 'timestamp'),
    'mActivity': (mActivity_df, 'timestamp'),
    'mAmbience': (mAmbience_df, 'timestamp'),
    'mBle': (mBle_df, 'timestamp'),
    'mGps': (mGps_df, 'timestamp'),
    'mLight': (mLight_df, 'timestamp'),
    'mScreenStatus': (mScreenStatus_df, 'timestamp'),
    'mUsageStats': (mUsageStats_df, 'timestamp'),
    'mWifi': (mWifi_df, 'timestamp'),
    'wHr': (wHr_df, 'timestamp'),
    'wLight': (wLight_df, 'timestamp'),
    'wPedo': (wPedo_df, 'timestamp'),
}

# %% [markdown]
# # 학습 데이터 분리

# %%
# ✅ 결과 저장
for name, (df, ts_col) in dataframes.items():
    print(f"⏳ {name} 분리 중...")
    test_df, train_df = split_test_train(df.copy(), subject_col='subject_id', timestamp_col=ts_col)
    globals()[f"{name}_test"] = test_df
    globals()[f"{name}_train"] = train_df
    print(f"✅ {name}_test → {test_df.shape}, {name}_train → {train_df.shape}")

# %% [markdown]
# # 5분 단위 평균 T=492
# * 만약 결측치가 5분이 넘어갈 경우 보간
# * 원래 lifelog_Data, Sleep_data(lisfelog_data+1) 둘 다 있으나 현재는 예측이기에 lifelog_data만 사용

# %%
modality_names = ['mACStatus', 'mActivity', 'mBle', 'mGps', 'mLight',
                  'mScreenStatus', 'mUsageStats', 'mWifi', 'wHr', 'wLight', 'wPedo']
for name in modality_names:   # ['BLE', 'HR', 'ACC', ...]
    train_df:pd.DataFrame = globals()[f"{name}_train"]
    os.makedirs(f"ETRI 2024/train", exist_ok=True)
    print(f"{name} info")
    print(train_df.info())
    print(train_df.head(2))
    print("\n")
    train_df.head(50).to_csv(f"ETRI 2024/train/{name}_train_sample.csv", index=False)

# %%
processed_dict = {}

modality_handlers = {
    "wHr": preprocess_wHr,
    "mGps": preprocess_mGps,
    "mWifi": preprocess_mWifi,
    "mUsageStats": preprocess_mUsage,
    "mBle": preprocess_mBle,
}



modality_names = ['mLight','mACStatus', 'mActivity', 'mBle', 'mGps', 'wHr',
                  'mScreenStatus', 'mUsageStats', 'mWifi', 'wLight', 'wPedo']
for name in modality_names:   # ['wHr', 'mBle', 'mWifi', ...]
    train_df = globals()[f"{name}_train"]

    
    print(f"⏳ {name} 전처리 중...")
    if name in modality_handlers.keys():
        preprocess_func = modality_handlers[name]
        train_df = preprocess_func(train_df)

    if not os.path.exists(f"ETRI 2024/train/{name}_train_preprocess_input.csv"):
        train_df.to_csv(f"ETRI 2024/train/{name}_train_preprocess_input.csv", index=False)
    proc = preprocess_all_days(
        df=train_df,
        metric_df=metric_train_df,
        resample_freq=FREQUENCY,
        interpolation=INTERPOLATION,
        min_ratio=MIN_RATIO,
        mask=True
    )

    processed_dict[name] = proc

# %%
processed_dict['mLight']

# %%
final_dataset = {}

keys = processed_dict["mACStatus"].keys()   # 공통 key

for key in keys:
    final_dataset[key] = {}
    for name in modality_names:
        if key in processed_dict[name]:
            final_dataset[key][name] = processed_dict[name][key] # KEY: MODALITY -> (subid, date)로 변환


# %%
file_name = f"ETRI 2024/processed_{INTERPOLATION}_{FREQUENCY}min_{MIN_RATIO*100:.0f}%_MIN_MASK_dataset.pkl"
import pickle
with open(file_name, "wb") as f:
    pickle.dump(final_dataset, f)

# %%
from collections import defaultdict

modality_names = processed_dict.keys()

# 모달리티별 missing count
missing_by_modality = defaultdict(int)

# 날짜별 모달리티 개수
modalities_per_day = defaultdict(int)

# 날짜 리스트
all_keys = list(final_dataset.keys())

for key in all_keys:
    day_modalities = final_dataset[key]
    
    count_present = 0
    for m in modality_names:
        if m in day_modalities and day_modalities[m] is not None:
            count_present += 1
        else:
            missing_by_modality[m] += 1

    modalities_per_day[key] = count_present

# ✔ 모든 모달리티가 있는 날짜
complete_days = [k for k, c in modalities_per_day.items() if c == len(modality_names)]

# ✔ 하나라도 부족한 날짜
incomplete_days = [k for k, c in modalities_per_day.items() if c < len(modality_names)]

# ✔ coverage percent
coverage = {m: 1 - missing_by_modality[m] / len(all_keys) for m in modality_names}


# ------------------ 출력 ------------------

print("📌 모달리티별 Missing 개수:")
for m in modality_names:
    print(f"  - {m}: {missing_by_modality[m]}개 missing")

print("\n📌 날짜별 모달리티 개수 (예: 5개 있으면 5개)")
for k, v in list(modalities_per_day.items())[:10]:  # 앞 10개만 미리보기
    print(f"{k}: {v}개")

print("\n📌 모든 모달리티가 있는 날짜 개수:", len(complete_days))
print("📌 하나라도 빠진 날짜 개수:", len(incomplete_days))

print("\n📌 모달리티별 coverage 비율 (%):")
for m in modality_names:
    print(f"  - {m}: {coverage[m]*100:.2f}%")



# %%

print_dict_structure(final_dataset, max_value_length=10)  # 앞 10개 키만 출력


