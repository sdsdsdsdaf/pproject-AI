import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import ast
try:
    from Utils.util import safe_to_list
except:
    from util import safe_to_list

def preprocess_mBle(df: pd.DataFrame):
    df = df.copy()
    df["m_ble"] = df["m_ble"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    def extract_ble(list_obj):
        list_obj = safe_to_list(list_obj)

        if not list_obj or len(list_obj) == 0:
            return pd.Series({
                "ble_count": 0,
                "ble_rssi_mean": np.nan,
                "ble_rssi_max": np.nan,
                "ble_rssi_min": np.nan,
                "ble_unique_address": 0,
                "ble_device_class_unique": 0,
            })

        # 리스트 속 dict에서 RSSI 추출
        rssi = [d.get("rssi", np.nan) for d in list_obj]

        # BLE address 개수
        addresses = [d.get("address", None) for d in list_obj]
        unique_addresses = len(set(addresses))

        # device_class 개수
        device_classes = [d.get("device_class", None) for d in list_obj]
        unique_device_class = len(set(device_classes))

        return pd.Series({
            "ble_count": len(list_obj),
            "ble_rssi_mean": np.nanmean(rssi),
            "ble_rssi_max": np.nanmax(rssi),
            "ble_rssi_min": np.nanmin(rssi),
            "ble_unique_address": unique_addresses,
            "ble_device_class_unique": unique_device_class,
        })

    # tqdm 적용 (index 유지)
    feats = pd.DataFrame(
        [extract_ble(x) for x in tqdm(df["m_ble"], desc="Extracting BLE features", leave=False)],
        index=df.index
    )

    df = pd.concat([df.drop(columns=["m_ble"]), feats], axis=1)
    return df


count = 0

def preprocess_wHr(df:pd.DataFrame):
    df = df.copy()
    df["heart_rate"] = df["heart_rate"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    global count
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

    # tqdm 적용
    feats = pd.DataFrame(
        [extract_hr(x) for x in tqdm(df["heart_rate"], desc="Extracting HR features", leave=False)],
        index=df.index
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
        [extract_gps(x) for x in tqdm(df["m_gps"], desc="Extracting GPS features", leave=False)],
        index=df.index
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
        [extract_wifi(x) for x in tqdm(df["m_wifi"], desc="Extracting WIFI features", leave=False)],
        index=df.index
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
        [extract_usage(x) for x in tqdm(df["m_usage_stats"], desc="Extracting usage features", leave=False)],
        index=df.index
    )
    df = pd.concat([df.drop(columns=["m_usage_stats"]), feats], axis=1)
    return df


def preprocess_day(
    df: pd.DataFrame,
    original_freq: float,
    resample_freq: float = 5.0,
    interpolation: str = "time",
    min_ratio: float = 0.5,
    mask: bool = False,
    target_date=None,
):
    """
    전처리 목적:
        하루 단위 시계열 데이터를 동일한 길이(T)를 가지도록 정규화한다.
        최종 출력은 resample_freq(예: 5분 단위)에 맞춘 일정 길이의 시계열이며,
        mask=True일 경우 '원본 값(0) / 보간으로 채워진 값(1)'을 구분하는 mask도 함께 반환한다.

    처리 순서:
    ---------------------------
    1) timestamp 인덱스 보장 및 numeric 컬럼만 선택
    2) original_freq 기반으로 하루 데이터 커버리지(valid_ratio) 계산
       - valid_ratio < min_ratio 이면 데이터 불충분 → None 반환
    3) full-day(00:00~24:00) 1분 단위 index 생성
    4) 원본 df를 full-day timeline 위에 reindex → aligned
    5) 업샘플(original_freq > resample_freq) 또는 다운샘플로 분기
       - 업샘플: target grid 로 reindex 후 보간(interp + ffill + bfill)
       - 다운샘플: resample.mean() 후 보간(interp + ffill + bfill)
    6) mask=True 일 경우:
       - 보간되기 전 NaN 위치(before_interp_nan)를 기반으로
         최종 day와 동일한 크기의 mask 생성(보간=1, 원본=0)

    Args:
        df (pd.DataFrame):
            한 날짜의 원본 데이터. timestamp 컬럼 또는 DatetimeIndex 필요.
        original_freq (float):
            원본 데이터의 sampling 간격(분 단위).
            업샘플/다운샘플을 결정하는 절대 기준.
        resample_freq (float, default=5.0):
            최종 출력 시계열의 간격(분 단위).
        interpolation (str, default="time"):
            pandas interpolate 방식.
        min_ratio (float, default=0.5):
            하루 중 '데이터가 존재한다고 간주되는 시간'의 최소 비율.
            유효시간 ≈ len(df) * original_freq 로 계산.
        mask (bool, default=False):
            True일 경우 보간 여부 mask(0=원본, 1=보간)를 함께 반환.
        target_date (datetime.date or None):
            full-day timeline 생성 기준 날짜. None이면 df 첫 timestamp 날짜.

    Returns:
        day : pd.DataFrame 또는 None
            resample_freq 간격으로 정규화된 하루 데이터.
        mask_df : pd.DataFrame 또는 None
            mask=True일 때만 반환. 0=원본, 1=보간.
            day 와 동일한 인덱스(shape)를 가진다.
    """

    # ----------------------------------------------------------
    # (0) timestamp 인덱스 보장 + numeric 데이터만 필터링
    # ----------------------------------------------------------
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("timestamp")

    
    df.index = pd.to_datetime(df.index).floor('min')
    df = df.sort_index()

    df = df.select_dtypes(include=["number"])
    if df.empty:
        return (None, None) if mask else None

    # ----------------------------------------------------------
    # (1) 원본 커버리지 계산 (len(df) * original_freq)
    # ----------------------------------------------------------
    day_minutes = 24 * 60
    covered_minutes = min(len(df) * original_freq, day_minutes)
    valid_ratio = covered_minutes / day_minutes

    # 커버리지 부족 → 데이터 불인정
    if valid_ratio < min_ratio:
        return (None, None) if mask else None

    # ----------------------------------------------------------
    # (2) full-day 1분 단위 index 생성
    # ----------------------------------------------------------
    if target_date is None:
        date_base = df.index[0].date()
    else:
        date_base = target_date

    start = pd.Timestamp(date_base)
    end = start + pd.Timedelta(days=1)

    # 00:00 ~ 23:59 총 1440 tick (pandas 특성상 마지막 제거)
    full_index = pd.date_range(start=start, end=end, freq="1min")[:-1]
 
    # full-day 1분 timeline 기준으로 원본 정렬
    aligned = df.reindex(full_index)
    # ----------------------------------------------------------
    # (3) 최종 resample grid 생성 (예: 5분 단위)
    # ----------------------------------------------------------
    target_index = pd.date_range(start=start, end=end, freq=f"{resample_freq}min")[:-1]

    # ----------------------------------------------------------
    # (4) 업샘플 or 다운샘플 분기
    # ----------------------------------------------------------

    # ==========================================================
    # 업샘플: original_freq > resample_freq
    #   ex) 원본 30분 간격 → 최종 5분 간격으로 확장
    #   mean() 절대 사용 금지 → 반드시 reindex + interpolate
    # ==========================================================
    if original_freq > resample_freq:

        # 먼저 target grid 로 선형 매핑

        day = aligned.ffill().bfill().reindex(target_index)
        day = day.resample(f"{resample_freq}min").ffill()

        # 보간 수행 (interpolate → ffill → bfill)
        numeric_cols = day.select_dtypes(include=["number"]).columns
        day_before_interp = day.copy()
        day = day.apply(pd.to_numeric, errors="coerce")

        if interpolation in ["linear", "time", "mean"]:
            day[numeric_cols] = (
                day[numeric_cols]
                .interpolate(interpolation)
                .ffill()
                .bfill()
            )
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")

        if day.isna().any().any():
            print("\n", day.isna().sum())
            print("Warning: NaN values remain after interpolation.")
            raise ValueError("NaN values remain after interpolation.")

        # mask 생성
        if mask:
            mask_df = day_before_interp.isna().astype(int)
            mask_df.index = day.index
            return day, mask_df

        return day

    # ==========================================================
    # 다운샘플: original_freq <= resample_freq
    #   ex) 원본 1분 간격 → 5분 평균
    #   resample.mean() 사용한 후 interpolate
    # ==========================================================
    else:

    # --------------------------
    # (A) 다운샘플: resample → mean
    # --------------------------

    # 먼저 resample하기 전에 구간 단위 mask 계산
    # resample된 각 time bucket(예: 5분)의 원본 데이터 존재 여부 확인
        mask_df = None
        if mask:
            # bucket 안에 원본 데이터가 하나도 없으면 1, 있으면 0
            mask_df = aligned.resample(f"{resample_freq}min") \
                            .apply(lambda df_win: df_win.isna().all().all()) \
                            .astype(int)

        # --------------------------
        # (B) resample → mean 적용
        # --------------------------
        day = aligned.resample(f"{resample_freq}min").mean()

        # --------------------------
        # (C) 보간 전 NaN 기록 (참고용)
        # --------------------------
        before_interp_nan = day.isna()

        # --------------------------
        # (D) 보간 수행
        # --------------------------
        numeric_cols = day.select_dtypes(include=["number"]).columns
        if interpolation in ["linear", "time", "mean"]:
            day[numeric_cols] = (
                day[numeric_cols]
                .interpolate(interpolation)
                .ffill()
                .bfill()
            )
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation}")
        
        if day.isna().any().any():
            print("\n", day.isna().sum())
            print("Warning: NaN values remain after interpolation.")
            raise ValueError("NaN values remain after interpolation.")

        # --------------------------
        # (E) mask 반환 (shape matched)
        # --------------------------
        if mask:
            mask_df.index = day.index   # resample index 동일
            return day, mask_df

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
                target_date=date,
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





