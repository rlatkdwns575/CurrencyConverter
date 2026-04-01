# ============================================================
# threshold_relative.py
# 저장된 Transformer 모델을 불러와
# "주간 Top-1 + 상대 순위 기반 매수" 방식으로 평가하는 독립 실행 파일
# ============================================================

import math
import pickle
import numpy as np
import pandas as pd
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

# ============================================================
# 0. 경로 및 설정
# ============================================================

RAW_DATA_PATH = "./data/raw_data.csv"
META_PATH = "./outputs/meta.pkl"
SCALER_PATH = "./outputs/scaler.pkl"
MODEL_PATH = "./outputs/transformer_model.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HOLD_DAYS = 126
TARGET_ANNUAL_RETURN = 0.07

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# nhead는 state_dict만으로 확정 복원이 어려우므로
# 학습 때 쓴 값으로 맞춰야 함
NHEAD = 4

# ---------------------------
# 상대 순위 기반 매수 설정
# ---------------------------
# 매주 Top-1 후보들 중,
# 그 후보군 전체에서 상위 몇 %만 살지 결정
# 예: q=0.70 -> 주간 후보들 중 상위 30%만 매수
QUANTILE_GRID = np.arange(0.50, 0.96, 0.05)

# 너무 낮은 확률은 후보에서 제거하기 위한 최소 floor
# 지금처럼 확률 스케일이 눌려있는 상황에서 매우 낮게 둠
PROB_FLOOR_GRID = [0.0, 0.005, 0.01, 0.015, 0.02]

# 거래 수가 너무 적으면 착시가 생기므로 최소 거래 수 제약
MIN_TRADES = 12


# ============================================================
# 1. 모델 정의
# ============================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


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

        logits = self.classifier(x).squeeze(-1)
        return logits


# ============================================================
# 2. 유틸 함수
# ============================================================

def load_and_prepare_data(path):
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


def make_sequences(df, feature_cols, seq_len):
    X, y, dates = [], [], []

    values = df[feature_cols].values
    targets = df["target"].values
    index = df.index

    for i in range(seq_len, len(df)):
        X.append(values[i - seq_len:i])
        y.append(targets[i])
        dates.append(index[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    dates = np.array(dates)

    return X, y, dates


def evaluate_binary_metrics(y_true, y_prob, y_pred):
    metrics = {
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    return metrics


def print_distribution_diagnostics(train_vis, valid_vis, test_vis):
    print("========== Probability Describe ==========")
    print("[Train]")
    print(train_vis["prob"].describe())

    print("\n[Valid]")
    print(valid_vis["prob"].describe())

    print("\n[Test]")
    print(test_vis["prob"].describe())

    print("\n========== Target Distribution ==========")
    print("[Train]")
    print(train_vis["target"].value_counts(dropna=False))
    print("positive ratio:", train_vis["target"].mean())

    print("\n[Valid]")
    print(valid_vis["target"].value_counts(dropna=False))
    print("positive ratio:", valid_vis["target"].mean())

    print("\n[Test]")
    print(test_vis["target"].value_counts(dropna=False))
    print("positive ratio:", test_vis["target"].mean())


def inspect_top_probs(df, top_n=20):
    cols = ["prob", "target", "usd_krw", "hold_return_6m", "ann_return_6m"]
    return df.sort_values("prob", ascending=False)[cols].head(top_n).copy()


# ============================================================
# 3. vis_df 생성
# ============================================================

def build_vis_df():
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]
    best_params = meta["best_params"]

    print("meta 로드 완료")
    print("feature 개수:", len(feature_cols))
    print("seq_len:", seq_len)

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

    print("\n모델 구조 추정값")
    print("input_dim        :", input_dim)
    print("d_model          :", d_model)
    print("dim_feedforward  :", dim_feedforward)
    print("num_layers       :", num_layers)
    print("nhead(수동입력)   :", NHEAD)

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

    print("\n모델 로드 완료")

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

    return vis_df


# ============================================================
# 4. 주간 Top-1 + 상대 순위 기반 매수
# ============================================================

def build_weekly_candidates(df):
    """
    각 주에서 prob가 가장 높은 하루 1개만 후보로 만든다.
    """
    out = df.copy()
    iso = out.index.isocalendar()
    out["year"] = iso.year.astype(int)
    out["week"] = iso.week.astype(int)

    idx = out.groupby(["year", "week"])["prob"].idxmax()
    candidates = out.loc[idx].copy().sort_index()

    return candidates


def apply_weekly_relative_rule(df, quantile_q=0.70, prob_floor=0.01):
    """
    1) 각 주의 최고 확률 하루만 후보
    2) 전체 후보군에서 quantile 기준 이상만 매수
    3) prob_floor 미만은 제거
    """
    out = df.copy()
    out["pred"] = 0

    candidates = build_weekly_candidates(out)

    # floor 적용
    candidates = candidates[candidates["prob"] >= prob_floor].copy()

    if len(candidates) == 0:
        return out, pd.DataFrame()

    # 분위수 threshold 계산
    relative_threshold = candidates["prob"].quantile(quantile_q)

    selected = candidates[candidates["prob"] >= relative_threshold].copy()

    if len(selected) > 0:
        out.loc[selected.index, "pred"] = 1

    return out, selected


def evaluate_relative_strategy(df, quantile_q=0.70, prob_floor=0.01):
    eval_df, selected_df = apply_weekly_relative_rule(
        df,
        quantile_q=quantile_q,
        prob_floor=prob_floor
    )

    y_true = eval_df["target"].values
    y_pred = eval_df["pred"].values
    y_prob = eval_df["prob"].values

    metrics = evaluate_binary_metrics(y_true, y_prob, y_pred)

    signal_df = eval_df[eval_df["pred"] == 1].copy()

    if len(signal_df) == 0:
        invest = {
            "n_trades": 0,
            "avg_hold_return": np.nan,
            "avg_ann_return": np.nan,
            "win_rate_7pct": np.nan,
            "below_3pct_rate": np.nan,
            "negative_rate": np.nan,
            "trades_per_week": 0.0,
        }
    else:
        num_weeks = max(1, eval_df.index.to_period("W").nunique())
        invest = {
            "n_trades": len(signal_df),
            "avg_hold_return": signal_df["hold_return_6m"].mean(),
            "avg_ann_return": signal_df["ann_return_6m"].mean(),
            "win_rate_7pct": (signal_df["ann_return_6m"] >= TARGET_ANNUAL_RETURN).mean(),
            "below_3pct_rate": (signal_df["ann_return_6m"] < 0.03).mean(),
            "negative_rate": (signal_df["ann_return_6m"] < 0.0).mean(),
            "trades_per_week": len(signal_df) / num_weeks,
        }

    return eval_df, selected_df, metrics, invest


def search_relative_rules(valid_df, quantile_grid, prob_floor_grid):
    rows = []

    for q in quantile_grid:
        for floor in prob_floor_grid:
            eval_df, selected_df, metrics, invest = evaluate_relative_strategy(
                valid_df,
                quantile_q=q,
                prob_floor=floor
            )

            rows.append({
                "quantile_q": q,
                "prob_floor": floor,
                "auc": metrics["auc"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "n_trades": invest["n_trades"],
                "avg_hold_return": invest["avg_hold_return"],
                "avg_ann_return": invest["avg_ann_return"],
                "win_rate_7pct": invest["win_rate_7pct"],
                "below_3pct_rate": invest["below_3pct_rate"],
                "negative_rate": invest["negative_rate"],
                "trades_per_week": invest["trades_per_week"],
            })

    return pd.DataFrame(rows)


def pick_best_relative_rule(result_df):
    """
    최종 추천 규칙:
    1) 거래가 너무 적지 않을 것
    2) 손실률 최소
    3) 3% 미만 비율 최소
    4) 평균 연환산 수익률 최대
    5) 승률 최대
    """
    temp = result_df.copy()

    constrained = temp[
        (temp["n_trades"] >= MIN_TRADES) &
        (temp["trades_per_week"] > 0.10) &
        (temp["trades_per_week"] <= 1.00)
    ].copy()

    if len(constrained) == 0:
        constrained = temp[temp["n_trades"] >= 8].copy()

    if len(constrained) == 0:
        constrained = temp.copy()

    constrained["freq_gap"] = (constrained["trades_per_week"] - 1.0).abs()

    row = constrained.sort_values(
        ["negative_rate", "below_3pct_rate", "avg_ann_return", "win_rate_7pct", "precision", "freq_gap"],
        ascending=[True, True, False, False, False, True]
    ).iloc[0]

    return row


# ============================================================
# 5. 결과 출력
# ============================================================

def print_result_block(name, df, quantile_q, prob_floor):
    eval_df, selected_df, metrics, invest = evaluate_relative_strategy(
        df,
        quantile_q=quantile_q,
        prob_floor=prob_floor
    )

    print(f"\n[{name}]")
    print(f"AUC             : {metrics['auc']:.4f}")
    print(f"Accuracy        : {metrics['accuracy']:.4f}")
    print(f"F1              : {metrics['f1']:.4f}")
    print(f"Precision       : {metrics['precision']:.4f}")
    print(f"Recall          : {metrics['recall']:.4f}")
    print(f"n_trades        : {invest['n_trades']}")
    print(f"avg_hold_return : {invest['avg_hold_return']}")
    print(f"avg_ann_return  : {invest['avg_ann_return']}")
    print(f"win_rate_7pct   : {invest['win_rate_7pct']}")
    print(f"below_3pct_rate : {invest['below_3pct_rate']}")
    print(f"negative_rate   : {invest['negative_rate']}")
    print(f"trades_per_week : {invest['trades_per_week']}")

    return eval_df, selected_df, metrics, invest


# ============================================================
# 6. 메인 실행
# ============================================================

def main():
    vis_df = build_vis_df()

    n = len(vis_df)
    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    train_vis = vis_df.iloc[:train_end].copy()
    valid_vis = vis_df.iloc[train_end:valid_end].copy()
    test_vis = vis_df.iloc[valid_end:].copy()

    overall_pred = np.zeros(len(vis_df))
    overall_metrics = evaluate_binary_metrics(
        vis_df["target"].values,
        vis_df["prob"].values,
        overall_pred
    )

    print("\n전체 구간 AUC 중심 성능")
    print({"auc": overall_metrics["auc"]})

    print("\n분할 크기")
    print("train:", train_vis.shape)
    print("valid:", valid_vis.shape)
    print("test :", test_vis.shape)

    print_distribution_diagnostics(train_vis, valid_vis, test_vis)

    print("\n========== Top Probabilities: Valid ==========")
    print(inspect_top_probs(valid_vis, top_n=20))

    print("\n========== Top Probabilities: Test ==========")
    print(inspect_top_probs(test_vis, top_n=20))

    # validation에서 상대 순위 규칙 탐색
    result_df = search_relative_rules(
        valid_vis,
        quantile_grid=QUANTILE_GRID,
        prob_floor_grid=PROB_FLOOR_GRID
    )

    best_rule = pick_best_relative_rule(result_df)

    print("\n[상대 순위 기반 규칙 후보 중 최종 추천]")
    print(best_rule)

    selected_q = float(best_rule["quantile_q"])
    selected_floor = float(best_rule["prob_floor"])

    print("\n최종 선택 규칙")
    print("quantile_q :", selected_q)
    print("prob_floor :", selected_floor)

    train_eval_df, train_selected_df, train_metrics, train_invest = print_result_block(
        "Train", train_vis, selected_q, selected_floor
    )
    valid_eval_df, valid_selected_df, valid_metrics, valid_invest = print_result_block(
        "Valid", valid_vis, selected_q, selected_floor
    )
    test_eval_df, test_selected_df, test_metrics, test_invest = print_result_block(
        "Test", test_vis, selected_q, selected_floor
    )

    # ---------------------------
    # 시각화
    # ---------------------------
    pivot_ann = result_df.pivot_table(
        index="quantile_q", columns="prob_floor", values="avg_ann_return"
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_ann, aspect="auto", origin="lower")
    plt.colorbar(label="Valid Avg Annualized Return")
    plt.xticks(range(len(pivot_ann.columns)), [f"{x:.3f}" for x in pivot_ann.columns], rotation=45)
    plt.yticks(range(len(pivot_ann.index)), [f"{x:.2f}" for x in pivot_ann.index])
    plt.xlabel("prob_floor")
    plt.ylabel("quantile_q")
    plt.title("Validation Avg Annualized Return Heatmap")
    plt.tight_layout()
    plt.show()

    pivot_neg = result_df.pivot_table(
        index="quantile_q", columns="prob_floor", values="negative_rate"
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(pivot_neg, aspect="auto", origin="lower")
    plt.colorbar(label="Valid Negative Rate")
    plt.xticks(range(len(pivot_neg.columns)), [f"{x:.3f}" for x in pivot_neg.columns], rotation=45)
    plt.yticks(range(len(pivot_neg.index)), [f"{x:.2f}" for x in pivot_neg.index])
    plt.xlabel("prob_floor")
    plt.ylabel("quantile_q")
    plt.title("Validation Negative Rate Heatmap")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(test_eval_df.index, test_eval_df["prob"], label="Predicted Probability")
    if len(test_selected_df) > 0:
        plt.scatter(test_selected_df.index, test_selected_df["prob"], marker="^", s=60, label="Buy Signal")
    plt.title("Test Probability with Weekly Top-1 + Relative Rank Buy Signals")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(test_eval_df.index, test_eval_df["usd_krw"], label="USD/KRW")
    if len(test_selected_df) > 0:
        plt.scatter(test_selected_df.index, test_selected_df["usd_krw"], marker="^", s=60, label="Buy Signal")
    plt.title("USD/KRW with Weekly Top-1 + Relative Rank Buy Signals (Test)")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 저장
    result_df.to_csv("./outputs/relative_rule_search_valid.csv", index=False, encoding="utf-8-sig")
    train_eval_df.to_csv("./outputs/train_eval_relative_rule.csv", encoding="utf-8-sig")
    valid_eval_df.to_csv("./outputs/valid_eval_relative_rule.csv", encoding="utf-8-sig")
    test_eval_df.to_csv("./outputs/test_eval_relative_rule.csv", encoding="utf-8-sig")

    summary_df = pd.DataFrame([
        {
            "split": "train",
            **train_metrics,
            **train_invest,
            "selected_quantile_q": selected_q,
            "selected_prob_floor": selected_floor,
        },
        {
            "split": "valid",
            **valid_metrics,
            **valid_invest,
            "selected_quantile_q": selected_q,
            "selected_prob_floor": selected_floor,
        },
        {
            "split": "test",
            **test_metrics,
            **test_invest,
            "selected_quantile_q": selected_q,
            "selected_prob_floor": selected_floor,
        },
    ])
    summary_df.to_csv("./outputs/summary_relative_rule.csv", index=False, encoding="utf-8-sig")

    print("\n저장 완료")
    print("- ./outputs/relative_rule_search_valid.csv")
    print("- ./outputs/train_eval_relative_rule.csv")
    print("- ./outputs/valid_eval_relative_rule.csv")
    print("- ./outputs/test_eval_relative_rule.csv")
    print("- ./outputs/summary_relative_rule.csv")


if __name__ == "__main__":
    main()