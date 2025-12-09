import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import ast
try:
    from Utils.util import safe_to_list
except:
    from util import safe_to_list

def preprocess_wHr(df:pd.DataFrame):
    df = df.copy()
    df["heart_rate"] = df["heart_rate"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    count = 0
    def extract_hr(x):
        
        x = safe_to_list(x)

        if not x or len(x) == 0:
            count += 1
            return pd.Series({
                "hr_mean": np.nan,
                "hr_min": np.nan,
                "hr_max": np.nan,
                "hr_std": np.nan,
            })
        arr = np.array(x)
        return pd.Series({
            "hr_mean": arr.mean(),
            "hr_min": arr.min(),
            "hr_max": arr.max(),
            "hr_std": arr.std(),
        })

    print("Heart Rate Missing Count Before:", count)
    # tqdm 적용
    feats = pd.DataFrame(
        [extract_hr(x) for x in tqdm(df["heart_rate"], desc="Extracting HR features", leave=False)]
    )

    df = pd.concat([df.drop(columns=["heart_rate"]), feats], axis=1)
    return df

def preprocess_mGps(df:pd.DataFrame):
    df = df.copy()
    df["m_gps"] = df["m_gps"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    def extract_gps(list_obj):
        # 1) None or NaN
        # 3) 빈 list
        list_obj = safe_to_list(list_obj)
        if len(list_obj) == 0:
            return pd.Series({
                "gps_count": 0,
                "gps_alt_mean": np.nan,
                "gps_alt_min": np.nan,
                "gps_alt_max": np.nan,
                "gps_speed_mean": np.nan,
                "gps_speed_max": np.nan,
                "gps_speed_min": np.nan,
            })
            
        alt  = [d.get("altitude", np.nan) for d in list_obj]
        lat  = [d.get("latitude", np.nan) for d in list_obj]
        lon  = [d.get("longitude", np.nan) for d in list_obj]
        spd  = [d.get("speed", np.nan) for d in list_obj]

        return pd.Series({
            "gps_count": len(list_obj),
            "gps_alt_mean": np.nanmean(alt),
            "gps_lat_mean": np.nanmean(lat),
            "gps_lon_mean": np.nanmean(lon),
            "gps_speed_mean": np.nanmean(spd),
            "gps_speed_max": np.nanmax(spd),
            "gps_speed_min": np.nanmin(spd),
        })

    # tqdm 적용
    feats = pd.DataFrame(
        [extract_gps(x) for x in tqdm(df["m_gps"], desc="Extracting GPS features", leave=False)]
    )

    df = pd.concat([df.drop(columns=["m_gps"]), feats], axis=1)
    return df

def preprocess_mWifi(df:pd.DataFrame):
    df = df.copy()
    df["m_wifi"] = df["m_wifi"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    def extract_wifi(list_obj):
        list_obj = safe_to_list(list_obj)
        if not list_obj or len(list_obj) == 0:
            return pd.Series({
                "wifi_count": 0,
                "wifi_rssi_mean": np.nan,
                "wifi_rssi_max": np.nan,
                "wifi_rssi_min": np.nan,
            })

        rssi = [d.get("rssi", np.nan) for d in list_obj]

        return pd.Series({
            "wifi_count": len(list_obj),
            "wifi_rssi_mean": np.nanmean(rssi),
            "wifi_rssi_max": np.nanmax(rssi),
            "wifi_rssi_min": np.nanmin(rssi),
        })

    # tqdm 적용
    feats = pd.DataFrame(
        [extract_wifi(x) for x in tqdm(df["m_wifi"], desc="Extracting WIFI features", leave=False)]
    )
    df = pd.concat([df.drop(columns=["m_wifi"]), feats], axis=1)
    return df

def preprocess_mUsage(df:pd.DataFrame):
    df = df.copy()
    df["m_usage_stats"] = df["m_usage_stats"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    def extract_usage(list_obj):
        list_obj = safe_to_list(list_obj)
        if not list_obj or len(list_obj) == 0:
            return pd.Series({
                "usage_count": 0,
                "usage_total_time": np.nan,
                "usage_max_time": np.nan,
            })

        times = [d.get("total_time", np.nan) for d in list_obj]

        return pd.Series({
            "usage_count": len(list_obj),
            "usage_total_time": np.nansum(times),
            "usage_max_time": np.nanmax(times),
        })

    # tqdm 적용
    feats = pd.DataFrame(
        [extract_usage(x) for x in tqdm(df["m_usage_stats"], desc="Extracting usage features", leave=False)]
    )
    df = pd.concat([df.drop(columns=["m_usage_stats"]), feats], axis=1)
    return df


def preprocess_day(df:pd.DataFrame, original_freq, resample_freq=5,interpolation="time", min_ratio=0.5, mask: bool = False):
    """
    하루 단위 시계열 데이터 전처리 함수
    Args:
        df (pd.DataFrame): 입력 데이터프레임, 'timestamp' 컬럼이 있어야 함
        resample_freq (int): 리샘플링 간격 (기본값: 5 - 5분 간격)
        interpolation (str): 보간 방법 (기본값: "time" - 시간 기반 선형 보간)
        min_ratio (float): 최소 유효 데이터 비율 (기본값: 0.5)
    Returns:
        df (pd.DataFrame or None): 전처리된 데이터프레임 또는 결측 비율이 높아 None 반환
    """
    interpolation = interpolation.lower()
    interpolation_methods = ["time", "linear", "mean"]
    if interpolation not in interpolation_methods:
        raise ValueError(f"Invalid interpolation method. Choose from {interpolation_methods}")
    
    # timestamp 인덱스 설정
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")

    df = df.select_dtypes(include=["number"])

    # -------------------------------
    # 리샘플 간격 사용자 설정
    # ------------------------------- 
    valid_ratio = df.notna().mean().mean()
    if valid_ratio < min_ratio:
        return None if not mask else (None, None)
    
    if original_freq <= resample_freq:
        day = df.resample(f"{resample_freq}min").mean()
    else:
        day = df.resample(f"{resample_freq}min").asfreq()

    # -------------------------------
    # 결측 비율 확인
    # -------------------------------


    # -------------------------------
    # mask 생성 (보간 전 NaN 위치 기록)
    # -------------------------------
    if mask:
        nan_mask = day.isna().astype(int)

    # -------------------------------
    # 보간
    # -------------------------------
    day = day.infer_objects(copy=False)
    if interpolation in ["time", "linear"]:
        num_cols = day.select_dtypes(include=["number"]).columns
        day[num_cols] = day[num_cols].interpolate(interpolation).bfill().ffill()
    elif interpolation == "mean":
        day = day.fillna(day.mean())
    else:
        raise NotImplementedError("Currently, only 'time' and 'mean' interpolation methods are implemented.")

    if mask:
        return day, nan_mask

    return day


def preprocess_all_days(
    df: pd.DataFrame,
    metric_df: pd.DataFrame,
    resample_freq: int = 5,
    interpolation: str = "time",
    min_ratio: float = 0.5,
    mask: bool = False
):
    """
    전체 기간에 대해 하루 단위 시계열 데이터를 전처리하는 함수.
    metric_df에 포함된 (subject_id, lifelog_date) 쌍을 기준으로
    하루의 원본 데이터를 잘라 preprocess_day()를 적용한다.

    Args:
        df (pd.DataFrame):
            원본 lifelog 데이터.
            필수 컬럼: ['subject_id', 'timestamp']

        metric_df (pd.DataFrame):
            처리해야 할 날짜 정보가 포함된 메트릭 데이터프레임.
            필수 컬럼: ['subject_id', 'lifelog_date']

        resample_freq (int):
            리샘플링 간격 (분 단위). 예: 1, 5, 10

        interpolation (str):
            보간 방식 ("time", "linear", "mean")

        min_ratio (float):
            하루 데이터에서 사용 가능한 값 비율(최소 허용 기준)

        mask (bool):
            True면 preprocess_day()에서 (data, mask)를 반환하도록 함.

    Returns:
        dict:
            mask=False → {(subject_id, lifelog_date): day_df}
            mask=True  → {(subject_id, lifelog_date): (day_df, mask_df)}
    """

    # -------------------------------
    # (1) timestamp 정규화 및 정렬
    # -------------------------------
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["subject_id", "timestamp"])

    # -------------------------------
    # (2) 메트릭 날짜 정규화
    # -------------------------------
    metric_df = metric_df.copy()
    metric_df["lifelog_date"] = pd.to_datetime(metric_df["lifelog_date"]).dt.date

    processed = {}

    # -------------------------------
    # (3) 날짜 단위로 반복 처리
    # -------------------------------
    for row in tqdm(metric_df.itertuples(index=False), total=len(metric_df)):
        subject_id = row.subject_id
        date = row.lifelog_date

        start = pd.Timestamp(date)
        end = start + pd.Timedelta(days=1)

        # 하루 데이터 추출
        day_df = df[(df["subject_id"] == subject_id) &
                    (df["timestamp"] >= start) &
                    (df["timestamp"] < end)]

        if day_df.empty:
            continue

        # 원본 sampling interval(분) 추정
        diffs = day_df["timestamp"].diff().dropna().dt.total_seconds() / 60
        if len(diffs) == 0:
            continue
        original_freq = int(round(diffs.median()))

  
        if mask:
            result, mask_df = preprocess_day(
                df=day_df,
                original_freq=original_freq,
                resample_freq=resample_freq,
                interpolation=interpolation,
                min_ratio=min_ratio,
                mask=True
            )

            if result is not None:
                processed[(subject_id, date)] = (result, mask_df)

        else:
            result = preprocess_day(
                df=day_df,
                original_freq=original_freq,
                resample_freq=resample_freq,
                interpolation=interpolation,
                min_ratio=min_ratio,
                mask=False
            )

            if result is not None:
                processed[(subject_id, date)] = result

    return processed





