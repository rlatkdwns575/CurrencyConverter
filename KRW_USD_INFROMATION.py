# ============================================
# USD/KRW 장기 투자 모델용 통합 데이터 수집기
# - yfinance + ECOS + FRED
# - 일별 기준 병합
# - 월/주/분기 데이터는 일별로 forward fill
# - raw_data.csv 저장
# ============================================

import os
import time as tm
import requests
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================
# 0. 사용자 설정
# ============================================

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

# 실제 키로 바꿔주세요.
ECOS_API_KEY = "YOUR_ECOS_KEY"
FRED_API_KEY = "YOUR_FRED_KEY"

SAVE_DIR = "./data"
RAW_SAVE_PATH = os.path.join(SAVE_DIR, "raw_data.csv")

# yfinance에서 사용할 시장 데이터 티커
YF_TICKERS = {
    "usd_krw": "KRW=X",
    "dxy": "DX-Y.NYB",
    "kospi": "^KS11",
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "vix": "^VIX",
    "gold": "GC=F",
    "wti": "CL=F",
    "copper": "HG=F",
    "us10y_proxy": "^TNX",   # Yahoo에서 주로 10Y yield proxy로 사용
    "us2y_proxy": "^IRX",    # 완벽한 2Y가 아닐 수 있으므로 proxy로 사용
}

# ------------------------------------------------
# ECOS 변수 설정
# 반드시 본인이 확인한 유효한 코드/항목코드로 수정하세요.
# 아래는 "구조" 예시입니다.
# ------------------------------------------------
# freq:
# D = 일, M = 월, Q = 분기, A = 연
#
# 예시:
# - current_account: 경상수지
# - cd_rate: CD(91일)
# - cpi_kr: 한국 CPI
#
# stat_code / item_code는 ECOS 코드검색에서 확인 후 입력 필요
ECOS_CONFIG = [
    {
        "name": "current_account",
        "stat_code": "301Y013",   # 예시
        "item_code": "000000",    # 실제 항목코드로 수정
        "freq": "M",
    },
    {
        "name": "cd_rate",
        "stat_code": "721Y001",   # 실제 통계표코드로 수정
        "item_code": "2010000",    # 실제 항목코드로 수정
        "freq": "M",
    },
    {
        "name": "cpi_kr",
        "stat_code": "901Y009",   # 실제 통계표코드로 수정
        "item_code": "0",    # 실제 항목코드로 수정
        "freq": "M",
    },
]

# ------------------------------------------------
# FRED 변수 설정
# series_id는 필요에 따라 교체 가능
# ------------------------------------------------
FRED_CONFIG = [
    {"name": "us_cpi", "series_id": "CPIAUCSL"},           # US CPI
    {"name": "us10y", "series_id": "DGS10"},               # 10Y Treasury
    {"name": "us2y", "series_id": "DGS2"},                 # 2Y Treasury
    {"name": "fed_funds", "series_id": "FEDFUNDS"},        # Effective Fed Funds
    {"name": "us_reer", "series_id": "RBKRBIS"},          # 예시: 한국 REER 계열 여부 확인 필요
    {"name": "us_unrate", "series_id": "UNRATE"},          # 실업률
    {"name": "us_indpro", "series_id": "INDPRO"},          # 산업생산
]

# RBKORBIS 같은 시리즈는 실제 원하는 의미와 맞는지 FRED에서 직접 재확인 권장
# 사용 전 series_id 점검 필요


# ============================================
# 1. 공통 유틸
# ============================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def standardize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def reindex_to_daily(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    daily_index = pd.date_range(start=start_date, end=end_date, freq="D")
    df = df.copy()
    df = standardize_date_index(df)
    df = df.reindex(daily_index)
    return df


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def print_headline(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


# ============================================
# 2. yfinance 수집
# ============================================

def fetch_yfinance_data(tickers, start_date, end_date):
    print_headline("[1] yfinance 수집 시작")
    series_list = []

    for col_name, ticker in tickers.items():
        try:
            tmp = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                repair=True,
                progress=False,
                threads=False,
            )

            if tmp is None or tmp.empty:
                print(f"  - {col_name:15s} 비어 있음")
                continue

            # yfinance 결과가 MultiIndex일 수도 있어서 방어 처리
            if isinstance(tmp.columns, pd.MultiIndex):
                # Close 레벨만 추출
                if "Close" in tmp.columns.get_level_values(0):
                    close_obj = tmp.xs("Close", axis=1, level=0)
                else:
                    print(f"  - {col_name:15s} Close 없음: {tmp.columns}")
                    continue

                # 단일 티커면 DataFrame -> Series로 변환
                if isinstance(close_obj, pd.DataFrame):
                    if close_obj.shape[1] == 1:
                        s = close_obj.iloc[:, 0].copy()
                    else:
                        # 혹시 여러 컬럼이면 현재 ticker 우선
                        if ticker in close_obj.columns:
                            s = close_obj[ticker].copy()
                        else:
                            s = close_obj.iloc[:, 0].copy()
                else:
                    s = close_obj.copy()

            else:
                # 일반 단일 컬럼 구조
                if "Close" in tmp.columns:
                    s = tmp["Close"].copy()
                elif "Adj Close" in tmp.columns:
                    s = tmp["Adj Close"].copy()
                else:
                    print(f"  - {col_name:15s} Close/Adj Close 없음: {list(tmp.columns)}")
                    continue

            s = pd.to_numeric(s, errors="coerce").dropna()
            s.index = pd.to_datetime(s.index).normalize()
            s.name = col_name

            series_list.append(s)
            print(f"  - {col_name:15s} 수집 완료: {len(s):,}개")
            tm.sleep(0.5)

        except Exception as e:
            print(f"  - {col_name:15s} 수집 실패: {type(e).__name__}: {e}")

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

# ============================================
# 3. ECOS 수집
# ============================================

def _ecos_date_key(start_date: str, end_date: str, freq: str) -> Tuple[str, str]:
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    if freq == "D":
        return start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d")
    if freq == "M":
        return start_dt.strftime("%Y%m"), end_dt.strftime("%Y%m")
    if freq == "Q":
        # ECOS 분기 형식은 종종 YYYYQn 형태를 사용
        s_q = f"{start_dt.year}Q{((start_dt.month - 1)//3) + 1}"
        e_q = f"{end_dt.year}Q{((end_dt.month - 1)//3) + 1}"
        return s_q, e_q
    if freq == "A":
        return start_dt.strftime("%Y"), end_dt.strftime("%Y")

    raise ValueError(f"지원하지 않는 ECOS freq: {freq}")


def _parse_ecos_time_value(x: str, freq: str) -> pd.Timestamp:
    if freq == "D":
        return pd.to_datetime(x, format="%Y%m%d")
    if freq == "M":
        return pd.to_datetime(x, format="%Y%m")
    if freq == "Q":
        # 예: 2024Q1
        year = int(x[:4])
        quarter = int(x[-1])
        month = quarter * 3
        return pd.Timestamp(year=year, month=month, day=1)
    if freq == "A":
        return pd.to_datetime(x, format="%Y")
    raise ValueError(f"지원하지 않는 ECOS freq: {freq}")


def fetch_single_ecos_series(
    api_key: str,
    stat_code: str,
    item_code: str,
    col_name: str,
    freq: str,
    start_date: str,
    end_date: str,
    lang: str = "kr",
) -> pd.DataFrame:
    """
    ECOS StatisticSearch 단일 시계열 수집
    """
    tm.sleep(1.5)
    if not api_key or api_key == "YOUR_ECOS_API_KEY":
        raise ValueError("ECOS_API_KEY가 설정되지 않았습니다.")

    start_key, end_key = _ecos_date_key(start_date, end_date, freq)

    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/"
        f"{api_key}/json/{lang}/1/10000/"
        f"{stat_code}/{freq}/{start_key}/{end_key}/{item_code}"
    )

    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "StatisticSearch" not in data:
        raise ValueError(f"ECOS 응답 오류: {data}")

    block = data["StatisticSearch"]
    rows = block.get("row", [])

    if not rows:
        raise ValueError(f"ECOS row가 비어 있습니다: {data}")

    df = pd.DataFrame(rows)
    if "TIME" not in df.columns or "DATA_VALUE" not in df.columns:
        raise ValueError(f"ECOS 필드 누락: {df.columns.tolist()}")

    df = df[["TIME", "DATA_VALUE"]].copy()
    df.columns = ["date", col_name]
    df[col_name] = safe_numeric(df[col_name])
    df["date"] = df["date"].astype(str).apply(lambda x: _parse_ecos_time_value(x, freq))
    df = df.dropna(subset=[col_name]).set_index("date").sort_index()

    return df


def fetch_all_ecos(config_list: List[Dict], start_date: str, end_date: str) -> pd.DataFrame:
    print_headline("[2] ECOS 수집 시작")
    results = []

    for cfg in config_list:
        name = cfg["name"]
        stat_code = cfg["stat_code"]
        item_code = cfg["item_code"]
        freq = cfg["freq"]

        try:
            print(f"  - [{name}] 수집 중 ...")
            df = fetch_single_ecos_series(
                api_key=ECOS_API_KEY,
                stat_code=stat_code,
                item_code=item_code,
                col_name=name,
                freq=freq,
                start_date=start_date,
                end_date=end_date,
            )
            results.append(df)
            print(f"    -> 성공: {len(df):,}개")
        except Exception as e:
            print(f"    -> 실패: {e}")

    if not results:
        print("  ! ECOS 수집 실패 - 빈 데이터프레임 반환")
        return pd.DataFrame()

    merged = pd.concat(results, axis=1)
    merged = standardize_date_index(merged)
    return merged


# ============================================
# 4. FRED 수집
# ============================================

def fetch_single_fred_series(
    api_key: str,
    series_id: str,
    col_name: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    FRED series observations 조회
    """
    tm.sleep(1.5)
    if not api_key or api_key == "YOUR_FRED_API_KEY":
        raise ValueError("FRED_API_KEY가 설정되지 않았습니다.")

    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if "observations" not in data:
        raise ValueError(f"FRED 응답 오류: {data}")

    obs = pd.DataFrame(data["observations"])
    if obs.empty:
        raise ValueError(f"FRED 관측값이 비어 있습니다: {series_id}")

    obs = obs[["date", "value"]].copy()
    obs.columns = ["date", col_name]
    obs["date"] = pd.to_datetime(obs["date"])
    obs[col_name] = pd.to_numeric(obs[col_name].replace(".", np.nan), errors="coerce")
    obs = obs.dropna(subset=[col_name]).set_index("date").sort_index()

    return obs


def fetch_all_fred(config_list: List[Dict], start_date: str, end_date: str) -> pd.DataFrame:
    print_headline("[3] FRED 수집 시작")
    results = []

    for cfg in config_list:
        name = cfg["name"]
        series_id = cfg["series_id"]

        try:
            print(f"  - [{name}] ({series_id}) 수집 중 ...")
            df = fetch_single_fred_series(
                api_key=FRED_API_KEY,
                series_id=series_id,
                col_name=name,
                start_date=start_date,
                end_date=end_date,
            )
            results.append(df)
            print(f"    -> 성공: {len(df):,}개")
        except Exception as e:
            print(f"    -> 실패: {e}")

    if not results:
        print("  ! FRED 수집 실패 - 빈 데이터프레임 반환")
        return pd.DataFrame()

    merged = pd.concat(results, axis=1)
    merged = standardize_date_index(merged)
    return merged


# ============================================
# 5. 병합 및 전처리
# ============================================

def merge_all_sources(
    yf_df: pd.DataFrame,
    ecos_df: pd.DataFrame,
    fred_df: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    print_headline("[4] 데이터 병합")

    dfs = []

    if yf_df is not None and not yf_df.empty:
        dfs.append(yf_df)

    if ecos_df is not None and not ecos_df.empty:
        ecos_daily = reindex_to_daily(ecos_df, start_date, end_date).ffill()
        dfs.append(ecos_daily)

    if fred_df is not None and not fred_df.empty:
        fred_daily = reindex_to_daily(fred_df, start_date, end_date).ffill()
        dfs.append(fred_daily)

    if not dfs:
        raise ValueError("병합할 데이터가 없습니다.")

    merged = pd.concat(dfs, axis=1)
    merged = standardize_date_index(merged)

    # 기준 인덱스: 환율 데이터가 있는 날짜 위주로 남기는 것이 일반적
    if "usd_krw" in merged.columns:
        merged = merged[merged["usd_krw"].notna()]

    # 숫자형 강제
    for c in merged.columns:
        merged[c] = safe_numeric(merged[c])

    # 금융시장 휴장일, 거시지표 발표일 차이 보정
    merged = merged.ffill().bfill()

    # 파생 기본변수
    merged = add_basic_derived_features(merged)

    print(f"최종 shape: {merged.shape}")
    return merged


def add_basic_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    최소한의 공통 파생변수
    """
    df = df.copy()

    # 로그 변환
    log_targets = [
        "usd_krw", "dxy", "kospi", "sp500", "nasdaq",
        "gold", "wti", "copper"
    ]
    for col in log_targets:
        if col in df.columns:
            df[f"ln_{col}"] = np.log(df[col].replace(0, np.nan))

    # 수익률
    ret_targets = ["usd_krw", "dxy", "kospi", "sp500", "vix", "gold", "wti", "copper"]
    for col in ret_targets:
        if col in df.columns:
            df[f"{col}_ret_1"] = df[col].pct_change(1)
            df[f"{col}_ret_5"] = df[col].pct_change(5)
            df[f"{col}_ret_20"] = df[col].pct_change(20)

    # 이동평균 / 이격도
    ma_targets = ["usd_krw", "dxy", "kospi", "sp500"]
    windows = [5, 20, 60]
    for col in ma_targets:
        if col in df.columns:
            for w in windows:
                ma = df[col].rolling(w).mean()
                df[f"{col}_ma_{w}"] = ma
                df[f"{col}_dev_ma_{w}"] = (df[col] - ma) / ma

    # 변동성
    vol_targets = ["usd_krw", "dxy", "kospi", "sp500", "vix"]
    for col in vol_targets:
        if col in df.columns:
            df[f"{col}_vol_10"] = df[col].pct_change().rolling(10).std()
            df[f"{col}_vol_20"] = df[col].pct_change().rolling(20).std()

    # 금리차
    if "us10y" in df.columns and "us2y" in df.columns:
        df["us_term_spread"] = df["us10y"] - df["us2y"]

    # 한국-미국 금리차 예시
    if "cd_rate" in df.columns and "fed_funds" in df.columns:
        df["kr_us_rate_diff"] = df["cd_rate"] - df["fed_funds"]

    return df


# ============================================
# 6. 저장 및 실행
# ============================================

def run_pipeline():
    ensure_dir(SAVE_DIR)

    print_headline("통합 데이터 수집 파이프라인 시작")
    print(f"수집 구간: {START_DATE} ~ {END_DATE}")

    # 1) yfinance
    try:
        yf_df = fetch_yfinance_data(YF_TICKERS, START_DATE, END_DATE)
    except Exception as e:
        print(f"[오류] yfinance 수집 실패: {e}")
        yf_df = pd.DataFrame()

    # 2) ECOS
    try:
        ecos_df = fetch_all_ecos(ECOS_CONFIG, START_DATE, END_DATE)
    except Exception as e:
        print(f"[오류] ECOS 수집 실패: {e}")
        ecos_df = pd.DataFrame()

    # 3) FRED
    try:
        fred_df = fetch_all_fred(FRED_CONFIG, START_DATE, END_DATE)
    except Exception as e:
        print(f"[오류] FRED 수집 실패: {e}")
        fred_df = pd.DataFrame()

    # 4) 병합
    merged = merge_all_sources(
        yf_df=yf_df,
        ecos_df=ecos_df,
        fred_df=fred_df,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # 5) 저장
    merged.to_csv(RAW_SAVE_PATH, encoding="utf-8-sig")
    print_headline("저장 완료")
    print(f"저장 경로: {RAW_SAVE_PATH}")
    print(merged.tail(10))

    return merged


if __name__ == "__main__":
    df = run_pipeline()
    df.to_csv(RAW_SAVE_PATH, encoding="utf-8-sig")