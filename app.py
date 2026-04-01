# app.py
import math
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

# =========================
# 기본 설정
# =========================
st.set_page_config(page_title="USD/KRW 투자 시각화", layout="wide")

RAW_DATA_PATH = "./data/raw_data.csv"
META_PATH = "./outputs/meta.pkl"
SCALER_PATH = "./outputs/scaler.pkl"
MODEL_PATH = "./outputs/transformer_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HOLD_DAYS = 126
TARGET_ANNUAL_RETURN = 0.07

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15

# 학습 때 사용한 nhead 값으로 맞춰야 함
NHEAD = 4


# =========================
# 모델 정의
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
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

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
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

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        return self.classifier(x).squeeze(-1)


# =========================
# 유틸
# =========================
def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

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


def make_sequences(df: pd.DataFrame, feature_cols: list[str], seq_len: int):
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
    return out.loc[idx].copy().sort_index()


def apply_weekly_relative_rule(
    df: pd.DataFrame,
    quantile_q: float,
    prob_floor: float,
):
    out = df.copy()
    out["pred"] = 0

    candidates = build_weekly_candidates(out)
    candidates = candidates[candidates["prob"] >= prob_floor].copy()

    if len(candidates) == 0:
        return out, pd.DataFrame(), np.nan

    relative_threshold = candidates["prob"].quantile(quantile_q)
    selected = candidates[candidates["prob"] >= relative_threshold].copy()

    if len(selected) > 0:
        out.loc[selected.index, "pred"] = 1

    return out, selected, float(relative_threshold)


def summarize_strategy(df: pd.DataFrame):
    signal_df = df[df["pred"] == 1].copy()

    if len(signal_df) == 0:
        return {
            "n_trades": 0,
            "avg_hold_return": np.nan,
            "avg_ann_return": np.nan,
            "win_rate_7pct": np.nan,
            "below_3pct_rate": np.nan,
            "negative_rate": np.nan,
            "trades_per_week": 0.0,
        }

    num_weeks = max(1, df.index.to_period("W").nunique())

    return {
        "n_trades": len(signal_df),
        "avg_hold_return": signal_df["hold_return_6m"].mean(),
        "avg_ann_return": signal_df["ann_return_6m"].mean(),
        "win_rate_7pct": (signal_df["ann_return_6m"] >= TARGET_ANNUAL_RETURN).mean(),
        "below_3pct_rate": (signal_df["ann_return_6m"] < 0.03).mean(),
        "negative_rate": (signal_df["ann_return_6m"] < 0.0).mean(),
        "trades_per_week": len(signal_df) / num_weeks,
    }


@st.cache_data
def build_vis_df():
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]
    best_params = meta["best_params"]

    df = load_and_prepare_data(RAW_DATA_PATH)
    df_raw = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols])

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    input_dim = state_dict["input_proj.weight"].shape[1]
    d_model = state_dict["input_proj.weight"].shape[0]
    dim_feedforward = state_dict["encoder.layers.0.linear1.weight"].shape[0]

    layer_indices = set()
    for k in state_dict.keys():
        if k.startswith("encoder.layers."):
            layer_indices.add(int(k.split(".")[2]))
    num_layers = len(layer_indices)

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

    X_all, y_all, dates_all = make_sequences(df, feature_cols, seq_len)
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

    return vis_df, meta, {
        "input_dim": input_dim,
        "d_model": d_model,
        "dim_feedforward": dim_feedforward,
        "num_layers": num_layers,
        "nhead": NHEAD,
    }


# =========================
# 앱 시작
# =========================
st.title("USD/KRW 투자 결과 시각화")
vis_df, meta, model_info = build_vis_df()

n = len(vis_df)
train_end = int(n * TRAIN_RATIO)
valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

train_vis = vis_df.iloc[:train_end].copy()
valid_vis = vis_df.iloc[train_end:valid_end].copy()
test_vis = vis_df.iloc[valid_end:].copy()

st.sidebar.header("규칙 설정")
quantile_q = st.sidebar.slider("상위 분위수 기준(q)", 0.50, 0.95, 0.50, 0.05)
prob_floor = st.sidebar.slider("최소 확률 floor", 0.0, 0.05, 0.0, 0.005)
split_choice = st.sidebar.selectbox("표시 구간", ["Train", "Valid", "Test"])

selected_map = {
    "Train": train_vis,
    "Valid": valid_vis,
    "Test": test_vis,
}
current_df = selected_map[split_choice]

# 전체 정보
st.subheader("모델/데이터 정보")
col1, col2, col3, col4 = st.columns(4)
col1.metric("feature 개수", len(meta["feature_cols"]))
col2.metric("seq_len", meta["seq_len"])
col3.metric("d_model", model_info["d_model"])
col4.metric("num_layers", model_info["num_layers"])

# 전체 구간 AUC
dummy_pred = np.zeros(len(vis_df))
overall_metrics = evaluate_binary_metrics(
    vis_df["target"].values,
    vis_df["prob"].values,
    dummy_pred,
)
st.write("전체 AUC:", overall_metrics["auc"])

# 현재 규칙 적용
eval_df, selected_df, relative_threshold = apply_weekly_relative_rule(
    current_df,
    quantile_q=quantile_q,
    prob_floor=prob_floor,
)

metrics = evaluate_binary_metrics(
    eval_df["target"].values,
    eval_df["prob"].values,
    eval_df["pred"].values,
)
invest = summarize_strategy(eval_df)

st.subheader(f"{split_choice} 성능 요약")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("AUC", f"{metrics['auc']:.4f}")
m2.metric("F1", f"{metrics['f1']:.4f}")
m3.metric("Precision", f"{metrics['precision']:.4f}")
m4.metric("Recall", f"{metrics['recall']:.4f}")
m5.metric("거래 수", f"{invest['n_trades']}")

m6, m7, m8, m9 = st.columns(4)
m6.metric("평균 연환산 수익률", f"{invest['avg_ann_return']:.4f}" if pd.notna(invest["avg_ann_return"]) else "nan")
m7.metric("7% 달성 비율", f"{invest['win_rate_7pct']:.4f}" if pd.notna(invest["win_rate_7pct"]) else "nan")
m8.metric("3% 미만 비율", f"{invest['below_3pct_rate']:.4f}" if pd.notna(invest["below_3pct_rate"]) else "nan")
m9.metric("음수 비율", f"{invest['negative_rate']:.4f}" if pd.notna(invest["negative_rate"]) else "nan")

st.caption(f"주간 후보 내부 상대 기준 threshold: {relative_threshold:.6f}" if pd.notna(relative_threshold) else "선택된 후보 없음")

# 확률 분포
st.subheader("확률 분포")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.hist(current_df["prob"], bins=30)
ax1.set_title(f"{split_choice} Probability Distribution")
ax1.set_xlabel("prob")
ax1.set_ylabel("count")
st.pyplot(fig1)

# 환율 + 매수 시점
st.subheader("환율과 매수 시점")
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(eval_df.index, eval_df["usd_krw"], label="USD/KRW")
if len(selected_df) > 0:
    ax2.scatter(selected_df.index, selected_df["usd_krw"], marker="^", s=60, label="Buy Signal")
ax2.legend()
ax2.set_title(f"{split_choice} USD/KRW with Buy Signals")
st.pyplot(fig2)

# 확률 + 매수 시점
st.subheader("예측 확률과 매수 시점")
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(eval_df.index, eval_df["prob"], label="Predicted Probability")
if len(selected_df) > 0:
    ax3.scatter(selected_df.index, selected_df["prob"], marker="^", s=60, label="Buy Signal")
ax3.legend()
ax3.set_title(f"{split_choice} Probability with Buy Signals")
st.pyplot(fig3)

# 상위 확률 샘플
st.subheader("상위 확률 샘플")
top_df = current_df.sort_values("prob", ascending=False)[
    ["prob", "target", "usd_krw", "hold_return_6m", "ann_return_6m"]
].head(20)
st.dataframe(top_df)

# 선택된 매수 시점 표
st.subheader("선택된 매수 시점")
if len(selected_df) > 0:
    st.dataframe(
        selected_df[["prob", "target", "usd_krw", "hold_return_6m", "ann_return_6m"]]
        .sort_index()
    )
else:
    st.info("현재 규칙으로는 선택된 매수 시점이 없습니다.")

# split별 한 번에 비교
st.subheader("Split별 요약 비교")
summary_rows = []
for name, df_ in [("Train", train_vis), ("Valid", valid_vis), ("Test", test_vis)]:
    eval_tmp, selected_tmp, _ = apply_weekly_relative_rule(
        df_,
        quantile_q=quantile_q,
        prob_floor=prob_floor,
    )
    metrics_tmp = evaluate_binary_metrics(
        eval_tmp["target"].values,
        eval_tmp["prob"].values,
        eval_tmp["pred"].values,
    )
    invest_tmp = summarize_strategy(eval_tmp)
    summary_rows.append({
        "split": name,
        "auc": metrics_tmp["auc"],
        "f1": metrics_tmp["f1"],
        "precision": metrics_tmp["precision"],
        "recall": metrics_tmp["recall"],
        "n_trades": invest_tmp["n_trades"],
        "avg_ann_return": invest_tmp["avg_ann_return"],
        "win_rate_7pct": invest_tmp["win_rate_7pct"],
        "below_3pct_rate": invest_tmp["below_3pct_rate"],
        "negative_rate": invest_tmp["negative_rate"],
        "trades_per_week": invest_tmp["trades_per_week"],
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df)