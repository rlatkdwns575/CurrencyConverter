import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ============================================================
# finalapp.py
# 입력 날짜 기준으로
# 1) 매수 / 매수 불가 판단
# 2) 매수 확률 표시
# 3) 해당 주에서 가장 좋은 매수일 추천
# 을 보여주는 Streamlit 앱
# ============================================================

# ----------------------------
# 기본 경로 설정
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw_data.csv"
META_PATH = BASE_DIR / "outputs" / "meta.pkl"
SCALER_PATH = BASE_DIR / "outputs" / "scaler.pkl"
MODEL_PATH = BASE_DIR / "outputs" / "transformer_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 학습 당시 사용한 투자 기준
HOLD_DAYS = 126
TARGET_ANNUAL_RETURN = 0.07

# 저장된 state_dict만으로는 nhead를 완전히 복원하기 어려워서 수동으로 둡니다.
# 학습 때 사용한 값과 같아야 합니다.
NHEAD = 4


# ============================================================
# 모델 정의
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x).squeeze(-1)


# ============================================================
# 데이터 / 예측용 유틸
# ============================================================
def load_and_prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 학습 시와 동일한 방식
    df = df.ffill().bfill()

    if "usd_krw" not in df.columns:
        raise ValueError("raw_data.csv에 'usd_krw' 컬럼이 없습니다.")

    future_fx = df["usd_krw"].shift(-HOLD_DAYS)
    hold_return = future_fx / df["usd_krw"] - 1.0
    ann_return = (1.0 + hold_return) ** (365.0 / HOLD_DAYS) - 1.0

    df["future_fx"] = future_fx
    df["hold_return_6m"] = hold_return
    df["ann_return_6m"] = ann_return
    df["target"] = (df["ann_return_6m"] >= TARGET_ANNUAL_RETURN).astype(int)

    df = df.dropna(subset=["future_fx", "ann_return_6m"]).copy()
    return df


def make_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int):
    X, y, dates = [], [], []
    values = df[feature_cols].values
    targets = df["target"].values
    index = df.index

    for i in range(seq_len, len(df)):
        X.append(values[i - seq_len : i])
        y.append(targets[i])
        dates.append(index[i])

    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32),
        np.array(dates),
    )


def evaluate_binary_metrics(y_true, y_prob, y_pred):
    return {
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def build_weekly_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    iso = out.index.isocalendar()
    out["year"] = iso.year.astype(int)
    out["week"] = iso.week.astype(int)

    idx = out.groupby(["year", "week"])["prob"].idxmax()
    candidates = out.loc[idx].copy().sort_index()
    return candidates


@st.cache_data(show_spinner=False)
def load_meta_and_data():
    if not META_PATH.exists():
        raise FileNotFoundError(f"메타 파일이 없습니다: {META_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"스케일러 파일이 없습니다: {SCALER_PATH}")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {RAW_DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")

    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    df = load_and_prepare_data(RAW_DATA_PATH)
    df_raw = df.copy()

    return meta, scaler, df, df_raw


@st.cache_resource(show_spinner=False)
def load_model(meta: dict):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    input_dim = state_dict["input_proj.weight"].shape[1]
    d_model = state_dict["input_proj.weight"].shape[0]
    dim_feedforward = state_dict["encoder.layers.0.linear1.weight"].shape[0]

    layer_indices = set()
    for k in state_dict.keys():
        if k.startswith("encoder.layers."):
            layer_indices.add(int(k.split(".")[2]))
    num_layers = len(layer_indices)

    best_params = meta["best_params"]

    model = TransformerClassifier(
        input_dim=input_dim,
        d_model=d_model,
        nhead=NHEAD,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=best_params["dropout"],
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    model_info = {
        "input_dim": input_dim,
        "d_model": d_model,
        "dim_feedforward": dim_feedforward,
        "num_layers": num_layers,
        "nhead": NHEAD,
    }
    return model, model_info


@st.cache_data(show_spinner=False)
def build_vis_df():
    meta, scaler, df, df_raw = load_meta_and_data()
    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]

    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])

    model, model_info = load_model(meta)

    X_all, y_all, dates_all = make_sequences(df_scaled, feature_cols, seq_len)
    X_tensor = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(X_tensor)
        probs_all = torch.sigmoid(logits).cpu().numpy()

    vis_df = pd.DataFrame({
        "date": dates_all,
        "prob": probs_all,
        "target": y_all,
    }).set_index("date")

    extra_cols = df_raw.loc[dates_all, ["usd_krw", "hold_return_6m", "ann_return_6m"]].copy()
    vis_df = vis_df.join(extra_cols, how="left")

    return vis_df, meta, model_info


# ============================================================
# 운영 규칙
# ============================================================
def get_effective_date(index: pd.DatetimeIndex, selected_date: pd.Timestamp) -> pd.Timestamp:
    # 사용자가 주말/휴일을 입력하면, 그 이전의 가장 가까운 사용 가능한 날짜로 맞춥니다.
    valid_index = index[index <= selected_date]
    if len(valid_index) == 0:
        return index.min()
    return valid_index.max()


def get_week_slice(df: pd.DataFrame, base_date: pd.Timestamp) -> pd.DataFrame:
    iso = df.index.isocalendar()
    year = int(base_date.isocalendar().year)
    week = int(base_date.isocalendar().week)
    mask = (iso.year == year) & (iso.week == week)
    return df.loc[mask].copy().sort_index()


def evaluate_date_rule(
    vis_df: pd.DataFrame,
    selected_date: pd.Timestamp,
    quantile_q: float,
    prob_floor: float,
    ranking_scope: str,
):
    effective_date = get_effective_date(vis_df.index, selected_date)

    week_df = get_week_slice(vis_df, effective_date)
    if len(week_df) == 0:
        raise ValueError("선택한 날짜의 주간 데이터가 없습니다.")

    # 해당 주 최고 확률일
    best_day = week_df["prob"].idxmax()
    best_prob = float(week_df.loc[best_day, "prob"])

    # 상대 기준을 만들 후보군
    weekly_candidates = build_weekly_candidates(vis_df)

    if ranking_scope == "선택일 이전 데이터 기준":
        weekly_candidates = weekly_candidates.loc[weekly_candidates.index <= effective_date].copy()

    # floor 적용 후 분위수 threshold
    filtered_candidates = weekly_candidates[weekly_candidates["prob"] >= prob_floor].copy()

    if len(filtered_candidates) == 0:
        relative_threshold = np.nan
        buy_allowed_for_week = False
    else:
        relative_threshold = float(filtered_candidates["prob"].quantile(quantile_q))
        buy_allowed_for_week = best_prob >= relative_threshold and best_prob >= prob_floor

    selected_prob = float(vis_df.loc[effective_date, "prob"])
    selected_is_best = effective_date == best_day

    if not buy_allowed_for_week:
        decision = "매수 불가"
        reason = "해당 주 최고 확률일도 현재 선택한 운영 기준(상대 순위 + floor)을 넘지 못했습니다."
    else:
        if selected_is_best:
            decision = "매수"
            reason = "입력한 날짜가 해당 주 최고 확률일이며, 운영 기준을 만족합니다."
        else:
            decision = "매수 불가"
            reason = "입력한 날짜는 해당 주의 최고 확률일이 아닙니다. 추천 매수일을 확인하세요."

    return {
        "effective_date": effective_date,
        "selected_prob": selected_prob,
        "best_day": best_day,
        "best_prob": best_prob,
        "relative_threshold": relative_threshold,
        "buy_allowed_for_week": buy_allowed_for_week,
        "selected_is_best": selected_is_best,
        "decision": decision,
        "reason": reason,
        "week_df": week_df,
        "filtered_candidates": filtered_candidates,
    }


# ============================================================
# 시각화 함수
# ============================================================
def plot_weekly_probability(week_df: pd.DataFrame, effective_date: pd.Timestamp, best_day: pd.Timestamp):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(week_df.index, week_df["prob"], marker="o", label="매수 확률")
    if effective_date in week_df.index:
        ax.scatter([effective_date], [week_df.loc[effective_date, "prob"]], s=120, marker="s", label="입력 기준일")
    if best_day in week_df.index:
        ax.scatter([best_day], [week_df.loc[best_day, "prob"]], s=150, marker="^", label="주간 추천일")
    ax.set_title("해당 주의 일별 매수 확률")
    ax.set_xlabel("날짜")
    ax.set_ylabel("확률")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_weekly_exchange_rate(week_df: pd.DataFrame, effective_date: pd.Timestamp, best_day: pd.Timestamp):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(week_df.index, week_df["usd_krw"], marker="o", label="USD/KRW")
    if effective_date in week_df.index:
        ax.scatter([effective_date], [week_df.loc[effective_date, "usd_krw"]], s=120, marker="s", label="입력 기준일")
    if best_day in week_df.index:
        ax.scatter([best_day], [week_df.loc[best_day, "usd_krw"]], s=150, marker="^", label="주간 추천일")
    ax.set_title("해당 주의 환율 흐름")
    ax.set_xlabel("날짜")
    ax.set_ylabel("환율")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_candidate_distribution(filtered_candidates: pd.DataFrame, relative_threshold: float, best_prob: float):
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(filtered_candidates) > 0:
        ax.hist(filtered_candidates["prob"], bins=20)
        ax.axvline(relative_threshold, linestyle="--", label=f"상대 기준선 = {relative_threshold:.4f}")
        ax.axvline(best_prob, linestyle=":", label=f"이번 주 최고 확률 = {best_prob:.4f}")
        ax.set_title("주간 후보 확률 분포")
        ax.set_xlabel("확률")
        ax.set_ylabel("빈도")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "후보 데이터가 없습니다.", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    return fig


# ============================================================
# Streamlit UI
# ============================================================
def main():
    st.set_page_config(page_title="USD/KRW 매수 판단 앱", layout="wide")
    st.title("USD/KRW 매수 판단 및 주간 추천일 앱")
    st.caption("입력한 날짜를 기준으로 매수/매수 불가를 표시하고, 해당 주의 추천 매수일을 보여줍니다.")

    try:
        vis_df, meta, model_info = build_vis_df()
    except Exception as e:
        st.error(f"데이터 또는 모델 로드 중 오류가 발생했습니다: {e}")
        st.stop()

    # 사이드바
    st.sidebar.header("입력 설정")
    min_date = vis_df.index.min().date()
    max_date = vis_df.index.max().date()

    selected_date = st.sidebar.date_input(
        "판단할 날짜",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
    )
    quantile_q = st.sidebar.slider("후보 중 상위 분위수 기준(q)", 0.50, 0.95, 0.50, 0.05)
    prob_floor = st.sidebar.slider("최소 확률 floor", 0.0, 0.05, 0.0, 0.005)
    ranking_scope = st.sidebar.selectbox(
        "상대 순위 기준 범위",
        ["전체 데이터 기준", "선택일 이전 데이터 기준"],
        index=1,
    )

    # 모델 정보
    with st.expander("모델 정보"):
        st.write({
            "feature_count": len(meta["feature_cols"]),
            "seq_len": meta["seq_len"],
            **model_info,
        })

    selected_ts = pd.Timestamp(selected_date)

    try:
        result = evaluate_date_rule(
            vis_df=vis_df,
            selected_date=selected_ts,
            quantile_q=quantile_q,
            prob_floor=prob_floor,
            ranking_scope=ranking_scope,
        )
    except Exception as e:
        st.error(f"판단 중 오류가 발생했습니다: {e}")
        st.stop()

    effective_date = result["effective_date"]
    week_df = result["week_df"]
    best_day = result["best_day"]

    # 핵심 결과
    st.subheader("판단 결과")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("입력 날짜", str(selected_ts.date()))
    col2.metric("실제 사용 기준일", str(effective_date.date()))
    col3.metric("입력일 매수 확률", f"{result['selected_prob']:.4f}")
    col4.metric("이번 주 최고 확률", f"{result['best_prob']:.4f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("판단", result["decision"])
    col6.metric("주간 추천일", str(best_day.date()))
    col7.metric(
        "상대 기준선",
        f"{result['relative_threshold']:.4f}" if pd.notna(result["relative_threshold"]) else "nan"
    )

    if result["decision"] == "매수":
        st.success(result["reason"])
    else:
        st.warning(result["reason"])

    # 해당 주 요약
    st.subheader("해당 주 추천 요약")
    weekly_buyable = "예" if result["buy_allowed_for_week"] else "아니오"
    st.write({
        "해당 주 전체적으로 매수 가능한 주인가": weekly_buyable,
        "입력 날짜가 주간 최고 확률일인가": "예" if result["selected_is_best"] else "아니오",
        "prob_floor": prob_floor,
        "quantile_q": quantile_q,
    })

    # 그래프
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(plot_weekly_probability(week_df, effective_date, best_day))
    with c2:
        st.pyplot(plot_weekly_exchange_rate(week_df, effective_date, best_day))

    st.pyplot(
        plot_candidate_distribution(
            result["filtered_candidates"],
            result["relative_threshold"],
            result["best_prob"],
        )
    )

    # 테이블
    st.subheader("해당 주 일별 데이터")
    display_week = week_df.copy()
    display_week["is_input_day"] = display_week.index == effective_date
    display_week["is_best_day"] = display_week.index == best_day
    st.dataframe(
        display_week[["prob", "target", "usd_krw", "hold_return_6m", "ann_return_6m", "is_input_day", "is_best_day"]]
    )

    # 최근 주간 추천일 리스트
    st.subheader("최근 주간 추천일 미리보기")
    weekly_candidates = build_weekly_candidates(vis_df).copy()
    threshold_now = (
        result["filtered_candidates"]["prob"].quantile(quantile_q)
        if len(result["filtered_candidates"]) > 0 else np.inf
    )
    weekly_candidates["selected_by_current_rule"] = (
        (weekly_candidates["prob"] >= prob_floor)
        & (weekly_candidates["prob"] >= threshold_now)
    )
    st.dataframe(
        weekly_candidates[["prob", "target", "usd_krw", "hold_return_6m", "ann_return_6m", "selected_by_current_rule"]]
        .tail(20)
    )


if __name__ == "__main__":
    main()
