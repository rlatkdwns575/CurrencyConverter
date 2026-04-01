# ============================================================
# USD/KRW Transformer 전체 코드
# - 데이터 로드
# - 타깃 생성
# - Transformer 학습
# - Optuna 탐색
# - 모델 저장
# - 저장 모델 로드
# - threshold 탐색
# - 주간 Top-1 매수 규칙 적용
# ============================================================

import os
import math
import random
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")

# ============================================================
# 0. 설정
# ============================================================

DATA_PATH = "./data/raw_data.csv"

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(OUTPUT_DIR, "transformer_model.pth")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler.pkl")
META_PATH = os.path.join(OUTPUT_DIR, "meta.pkl")

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 투자 조건
HOLD_DAYS = 126  # 약 6개월(거래일)
TARGET_ANNUAL_RETURN = 0.07  # 연환산 7% 이상이면 target=1

# 학습 조건
N_TRIALS = 20
EPOCHS_DEFAULT = 30
PATIENCE = 7

# 데이터 분할
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# threshold 탐색
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)

# 주간 거래 빈도 목표
TARGET_TRADES_PER_WEEK = 1.0
MIN_TRADES_PER_WEEK = 0.15
MAX_TRADES_PER_WEEK = 1.20

# 손실 가중치 강도
# 이전보다 약하게 설정
NEGATIVE_WEIGHT = 2.0  # 음수 수익률
LOW3_WEIGHT = 1.3  # 0% ~ 3%
MID7_WEIGHT = 1.1  # 3% ~ 7%

# pos_weight 사용 여부
USE_POS_WEIGHT = False


# ============================================================
# 1. 재현성
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2. 데이터 로드 / 전처리
# ============================================================

def load_and_prepare_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig", index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 결측치 처리
    df = df.ffill().bfill()

    if "usd_krw" not in df.columns:
        raise ValueError("raw_data.csv에 'usd_krw' 컬럼이 없습니다.")

    # 6개월 뒤 환율
    future_fx = df["usd_krw"].shift(-HOLD_DAYS)

    # 6개월 단순 수익률
    hold_return = future_fx / df["usd_krw"] - 1.0

    # 연환산 수익률
    ann_return = (1.0 + hold_return) ** (365.0 / HOLD_DAYS) - 1.0

    df["future_fx"] = future_fx
    df["hold_return_6m"] = hold_return
    df["ann_return_6m"] = ann_return
    df["target"] = (df["ann_return_6m"] >= TARGET_ANNUAL_RETURN).astype(int)

    # 미래값 없는 마지막 구간 제거
    df = df.dropna(subset=["future_fx", "ann_return_6m"]).copy()

    return df


def select_feature_columns(df):
    exclude_cols = {
        "future_fx",
        "hold_return_6m",
        "ann_return_6m",
        "target",
    }

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    if len(feature_cols) == 0:
        raise ValueError("사용 가능한 feature column이 없습니다.")

    return feature_cols


def split_dataframe(df):
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_df = df.iloc[:n_train].copy()
    valid_df = df.iloc[n_train:n_train + n_valid].copy()
    test_df = df.iloc[n_train + n_valid:].copy()

    return train_df, valid_df, test_df


def get_split_boundaries(df):
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)

    train_end_date = df.index[n_train - 1]
    valid_end_date = df.index[n_train + n_valid - 1]

    return train_end_date, valid_end_date


# ============================================================
# 3. 시퀀스 생성
# ============================================================

def make_sequences(df, feature_cols, seq_len):
    X, y, ann_returns, dates = [], [], [], []

    values = df[feature_cols].values
    targets = df["target"].values
    ann_vals = df["ann_return_6m"].values
    index = df.index

    for i in range(seq_len, len(df)):
        X.append(values[i - seq_len:i])
        y.append(targets[i])
        ann_returns.append(ann_vals[i])
        dates.append(index[i])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    ann_returns = np.array(ann_returns, dtype=np.float32)
    dates = np.array(dates)

    return X, y, ann_returns, dates


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, ann_returns):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.ann_returns = torch.tensor(ann_returns, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.ann_returns[idx]


def build_dataloaders_from_full_df(df, feature_cols, seq_len, batch_size):
    X_all, y_all, ann_all, d_all = make_sequences(df, feature_cols, seq_len)

    train_end_date, valid_end_date = get_split_boundaries(df)

    train_mask = d_all <= train_end_date
    valid_mask = (d_all > train_end_date) & (d_all <= valid_end_date)
    test_mask = d_all > valid_end_date

    X_train, y_train, ann_train, d_train = X_all[train_mask], y_all[train_mask], ann_all[train_mask], d_all[train_mask]
    X_valid, y_valid, ann_valid, d_valid = X_all[valid_mask], y_all[valid_mask], ann_all[valid_mask], d_all[valid_mask]
    X_test, y_test, ann_test, d_test = X_all[test_mask], y_all[test_mask], ann_all[test_mask], d_all[test_mask]

    if len(X_train) == 0 or len(X_valid) == 0 or len(X_test) == 0:
        raise ValueError(
            f"시퀀스 부족: seq_len={seq_len}, train={len(X_train)}, valid={len(X_valid)}, test={len(X_test)}"
        )

    train_ds = TimeSeriesDataset(X_train, y_train, ann_train)
    valid_ds = TimeSeriesDataset(X_valid, y_valid, ann_valid)
    test_ds = TimeSeriesDataset(X_test, y_test, ann_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "X_train": X_train, "y_train": y_train, "ann_train": ann_train, "d_train": d_train,
        "X_valid": X_valid, "y_valid": y_valid, "ann_valid": ann_valid, "d_valid": d_valid,
        "X_test": X_test, "y_test": y_test, "ann_test": ann_test, "d_test": d_test,
    }


# ============================================================
# 4. 모델
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

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
# 5. 평가 / 손실 함수
# ============================================================

def get_class_pos_weight(y):
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    if pos == 0:
        return 1.0
    return float(neg / max(pos, 1))


def make_sample_weights(ann_returns):
    """
    샘플별 손실 가중치
    - 음수 수익률: 강하게 벌점
    - 0% ~ 3%: 중간 벌점
    - 3% ~ 7%: 약한 벌점
    - 7% 이상: 기본 가중치
    """
    weights = torch.ones_like(ann_returns)

    weights = torch.where(
        ann_returns < 0.0,
        torch.tensor(NEGATIVE_WEIGHT, device=ann_returns.device),
        weights
    )

    weights = torch.where(
        (ann_returns >= 0.0) & (ann_returns < 0.03),
        torch.tensor(LOW3_WEIGHT, device=ann_returns.device),
        weights
    )

    weights = torch.where(
        (ann_returns >= 0.03) & (ann_returns < 0.07),
        torch.tensor(MID7_WEIGHT, device=ann_returns.device),
        weights
    )

    return weights


def evaluate_binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    return metrics, y_pred


def run_one_epoch(model, loader, criterion, optimizer=None):
    if len(loader.dataset) == 0:
        raise ValueError("빈 dataset이 들어왔습니다.")

    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_probs = []
    all_targets = []

    for X_batch, y_batch, ann_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        ann_batch = ann_batch.to(DEVICE)

        with torch.set_grad_enabled(is_train):
            logits = model(X_batch)

            base_loss = criterion(logits, y_batch)
            sample_weights = make_sample_weights(ann_batch)
            loss = (base_loss * sample_weights).mean()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = y_batch.detach().cpu().numpy()

        total_loss += loss.item() * len(X_batch)
        all_probs.extend(probs.tolist())
        all_targets.extend(targets.tolist())

    avg_loss = total_loss / len(loader.dataset)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    metrics, _ = evaluate_binary_metrics(all_targets, all_probs)
    return avg_loss, metrics, all_probs, all_targets


def train_model(
        model,
        train_loader,
        valid_loader,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=30,
        patience=7,
        pos_weight=1.0,
        trial=None,
):
    # pos_weight는 기본적으로 끔
    if USE_POS_WEIGHT:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE),
            reduction="none"
        )
    else:
        criterion = nn.BCEWithLogitsLoss(
            reduction="none"
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_valid_auc = -np.inf
    best_epoch = 0

    history = {
        "train_loss": [],
        "valid_loss": [],
        "train_auc": [],
        "valid_auc": [],
    }

    wait = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_metrics, _, _ = run_one_epoch(model, train_loader, criterion, optimizer)
        valid_loss, valid_metrics, _, _ = run_one_epoch(model, valid_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["train_auc"].append(train_metrics["auc"])
        history["valid_auc"].append(valid_metrics["auc"])

        current_valid_auc = valid_metrics["auc"]

        if trial is not None:
            trial.report(current_valid_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if current_valid_auc > best_valid_auc:
            best_valid_auc = current_valid_auc
            best_epoch = epoch
            wait = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_valid_auc, best_epoch


# ============================================================
# 6. Optuna 목적 함수
# ============================================================

def create_objective(full_df, feature_cols):
    def objective(trial):
        d_model = trial.suggest_categorical("d_model", [32, 64, 96, 128])
        nhead_candidates = [h for h in [2, 4, 8] if d_model % h == 0]
        nhead = trial.suggest_categorical("nhead", nhead_candidates)
        num_layers = trial.suggest_int("num_layers", 1, 4)
        dim_feedforward = trial.suggest_categorical("dim_feedforward", [64, 128, 256, 512])
        dropout = trial.suggest_float("dropout", 0.05, 0.30)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
        seq_len = trial.suggest_categorical("seq_len", [120, 180, 252])

        try:
            loaders = build_dataloaders_from_full_df(
                full_df, feature_cols, seq_len, batch_size
            )
        except ValueError as e:
            raise optuna.exceptions.TrialPruned(str(e))

        pos_weight = get_class_pos_weight(loaders["y_train"])

        model = TransformerClassifier(
            input_dim=len(feature_cols),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        ).to(DEVICE)

        model, history, best_valid_auc, best_epoch = train_model(
            model=model,
            train_loader=loaders["train_loader"],
            valid_loader=loaders["valid_loader"],
            lr=lr,
            weight_decay=weight_decay,
            epochs=25,
            patience=5,
            pos_weight=pos_weight,
            trial=trial,
        )

        return best_valid_auc

    return objective


def train_best_model(full_df, feature_cols, best_params):
    seq_len = best_params["seq_len"]
    batch_size = best_params["batch_size"]

    loaders = build_dataloaders_from_full_df(full_df, feature_cols, seq_len, batch_size)
    pos_weight = get_class_pos_weight(loaders["y_train"])

    model = TransformerClassifier(
        input_dim=len(feature_cols),
        d_model=best_params["d_model"],
        nhead=best_params["nhead"],
        num_layers=best_params["num_layers"],
        dim_feedforward=best_params["dim_feedforward"],
        dropout=best_params["dropout"],
    ).to(DEVICE)

    model, history, best_valid_auc, best_epoch = train_model(
        model=model,
        train_loader=loaders["train_loader"],
        valid_loader=loaders["valid_loader"],
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        epochs=EPOCHS_DEFAULT,
        patience=PATIENCE,
        pos_weight=pos_weight,
        trial=None,
    )

    if USE_POS_WEIGHT:
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE),
            reduction="none"
        )
    else:
        criterion = nn.BCEWithLogitsLoss(reduction="none")

    train_loss, train_metrics, train_probs, train_true = run_one_epoch(
        model, loaders["train_loader"], criterion, optimizer=None
    )
    valid_loss, valid_metrics, valid_probs, valid_true = run_one_epoch(
        model, loaders["valid_loader"], criterion, optimizer=None
    )
    test_loss, test_metrics, test_probs, test_true = run_one_epoch(
        model, loaders["test_loader"], criterion, optimizer=None
    )

    result = {
        "model": model,
        "history": history,
        "best_valid_auc": best_valid_auc,
        "best_epoch": best_epoch,
        "train_metrics": train_metrics,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "train_probs": train_probs,
        "valid_probs": valid_probs,
        "test_probs": test_probs,
        "train_true": train_true,
        "valid_true": valid_true,
        "test_true": test_true,
        "train_dates": loaders["d_train"],
        "valid_dates": loaders["d_valid"],
        "test_dates": loaders["d_test"],
        "seq_len": seq_len,
        "best_params": best_params,
    }
    return result


# ============================================================
# 7. 시각화
# ============================================================

def plot_history(history):
    plt.figure(figsize=(10, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["valid_loss"], label="valid_loss")
    plt.title("Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(history["train_auc"], label="train_auc")
    plt.plot(history["valid_auc"], label="valid_auc")
    plt.title("AUC History")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. 저장 / 로드
# ============================================================

def save_artifacts(result, scaler, feature_cols):
    torch.save(result["model"].state_dict(), MODEL_PATH)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    meta = {
        "feature_cols": feature_cols,
        "seq_len": result["seq_len"],
        "best_params": result["best_params"],
    }
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print("모델 저장 완료")
    print(MODEL_PATH)
    print(SCALER_PATH)
    print(META_PATH)


def load_saved_model():
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    feature_cols = meta["feature_cols"]
    seq_len = meta["seq_len"]
    best_params = meta["best_params"]

    model = TransformerClassifier(
        input_dim=len(feature_cols),
        d_model=best_params["d_model"],
        nhead=best_params["nhead"],
        num_layers=best_params["num_layers"],
        dim_feedforward=best_params["dim_feedforward"],
        dropout=best_params["dropout"],
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    return model, scaler, feature_cols, seq_len, best_params


# ============================================================
# 9. 저장 모델 기반 예측 DataFrame 생성
# ============================================================

def build_visualization_df():
    model, scaler, feature_cols, seq_len, best_params = load_saved_model()

    df = load_and_prepare_data(DATA_PATH)
    df_raw = df.copy()

    # scaler는 train 기준으로 fit된 것이 저장돼 있으므로 그대로 사용
    df[feature_cols] = scaler.transform(df[feature_cols])

    X_all, y_all, ann_all, dates_all = make_sequences(df, feature_cols, seq_len)
    X_tensor = torch.tensor(X_all, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits = model(X_tensor)
        probs_all = torch.sigmoid(logits).cpu().numpy()

    vis_df = pd.DataFrame({
        "date": dates_all,
        "prob": probs_all,
        "target": y_all,
    }).set_index("date")

    # 여기에는 스케일링 전 원본 값 붙임
    extra_cols = df_raw.loc[dates_all, ["usd_krw", "hold_return_6m", "ann_return_6m"]].copy()
    vis_df = vis_df.join(extra_cols, how="left")

    return vis_df, best_params


# ============================================================
# 10. 주간 Top-1 규칙
# ============================================================

def apply_weekly_top1_rule(df, threshold):
    out = df.copy()
    out["raw_pred"] = (out["prob"] >= threshold).astype(int)
    out["pred"] = 0

    candidates = out[out["raw_pred"] == 1].copy()
    if len(candidates) == 0:
        return out

    iso = candidates.index.isocalendar()
    candidates["year"] = iso.year.astype(int)
    candidates["week"] = iso.week.astype(int)

    idx = candidates.groupby(["year", "week"])["prob"].idxmax()
    out.loc[idx, "pred"] = 1
    return out


def evaluate_df_with_threshold(df, threshold, weekly_top1=True):
    if weekly_top1:
        eval_df = apply_weekly_top1_rule(df, threshold)
    else:
        eval_df = df.copy()
        eval_df["pred"] = (eval_df["prob"] >= threshold).astype(int)

    y_true = eval_df["target"].values
    y_pred = eval_df["pred"].values
    y_prob = eval_df["prob"].values

    metrics = {
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }

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
            "win_rate_7pct": (signal_df["ann_return_6m"] >= 0.07).mean(),
            "below_3pct_rate": (signal_df["ann_return_6m"] < 0.03).mean(),
            "negative_rate": (signal_df["ann_return_6m"] < 0.0).mean(),
            "trades_per_week": len(signal_df) / num_weeks,
        }

    return eval_df, metrics, invest


def search_thresholds(valid_df, threshold_grid, weekly_top1=True):
    rows = []

    for th in threshold_grid:
        eval_df, metrics, invest = evaluate_df_with_threshold(
            valid_df, threshold=th, weekly_top1=weekly_top1
        )

        row = {
            "threshold": th,
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
        }
        rows.append(row)

    return pd.DataFrame(rows)


def pick_best_threshold_risk_aware_weekly(threshold_df):
    temp = threshold_df.copy()

    constrained = temp[
        (temp["trades_per_week"] >= MIN_TRADES_PER_WEEK) &
        (temp["trades_per_week"] <= MAX_TRADES_PER_WEEK) &
        (temp["n_trades"] >= 12)
        ].copy()

    if len(constrained) == 0:
        temp["freq_gap"] = (temp["trades_per_week"] - TARGET_TRADES_PER_WEEK).abs()
        constrained = temp.sort_values(["freq_gap", "negative_rate", "avg_ann_return"]).head(30).copy()

    if "freq_gap" not in constrained.columns:
        constrained["freq_gap"] = (constrained["trades_per_week"] - TARGET_TRADES_PER_WEEK).abs()

    row = constrained.sort_values(
        ["negative_rate", "below_3pct_rate", "freq_gap", "avg_ann_return", "win_rate_7pct", "precision"],
        ascending=[True, True, True, False, False, False]
    ).iloc[0]

    return row


# ============================================================
# 11. 진단 출력
# ============================================================

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
# 12. 메인 실행
# ============================================================

def train_and_save():
    print("=" * 80)
    print("1) 데이터 로드")
    print("=" * 80)
    df = load_and_prepare_data(DATA_PATH)

    feature_cols = select_feature_columns(df)

    print(f"전체 데이터 shape: {df.shape}")
    print(f"feature 개수: {len(feature_cols)}")
    print(f"양성 비율(target=1): {df['target'].mean():.4f}")

    train_df, valid_df, test_df = split_dataframe(df)

    scaler = StandardScaler()
    df_scaled = df.copy()
    scaler.fit(train_df[feature_cols])
    df_scaled[feature_cols] = scaler.transform(df[feature_cols])

    print("=" * 80)
    print("2) Optuna 탐색")
    print("=" * 80)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    objective = create_objective(df_scaled, feature_cols)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n[Best Trial]")
    print("Best Value (Valid AUC):", study.best_value)
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    result = train_best_model(df_scaled, feature_cols, study.best_params)

    print("\n[Train Metrics]")
    print(result["train_metrics"])

    print("\n[Valid Metrics]")
    print(result["valid_metrics"])

    print("\n[Test Metrics]")
    print(result["test_metrics"])

    save_artifacts(result, scaler, feature_cols)

    pd.DataFrame(study.trials_dataframe()).to_csv(
        os.path.join(OUTPUT_DIR, "optuna_trials.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    plot_history(result["history"])


def evaluate_saved_model():
    print("=" * 80)
    print("저장된 모델 평가 시작")
    print("=" * 80)

    vis_df, best_params = build_visualization_df()

    all_metrics, _ = evaluate_binary_metrics(
        vis_df["target"].values,
        vis_df["prob"].values,
        threshold=0.5
    )

    print("\n전체 구간 성능 (threshold=0.5)")
    print(all_metrics)

    n = len(vis_df)
    train_end = int(n * TRAIN_RATIO)
    valid_end = int(n * (TRAIN_RATIO + VALID_RATIO))

    train_vis = vis_df.iloc[:train_end].copy()
    valid_vis = vis_df.iloc[train_end:valid_end].copy()
    test_vis = vis_df.iloc[valid_end:].copy()

    print("\n분할 크기")
    print("train:", train_vis.shape)
    print("valid:", valid_vis.shape)
    print("test :", test_vis.shape)

    print_distribution_diagnostics(train_vis, valid_vis, test_vis)

    print("\n========== Top Probabilities: Valid ==========")
    print(inspect_top_probs(valid_vis, top_n=20))

    print("\n========== Top Probabilities: Test ==========")
    print(inspect_top_probs(test_vis, top_n=20))

    threshold_df = search_thresholds(
        valid_vis,
        threshold_grid=THRESHOLD_GRID,
        weekly_top1=True
    )

    threshold_df.to_csv(
        os.path.join(OUTPUT_DIR, "threshold_search_valid_weekly.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    best_risk_aware = pick_best_threshold_risk_aware_weekly(threshold_df)

    print("\n[Threshold 후보 - 추천: 위험 회피 + 주 1회 제약 기준]")
    print(best_risk_aware)

    selected_threshold = float(best_risk_aware["threshold"])
    print("\n최종 추천 threshold:", selected_threshold)

    def print_result_block(name, df, threshold):
        eval_df, metrics, invest = evaluate_df_with_threshold(df, threshold=threshold, weekly_top1=True)

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

        return eval_df, metrics, invest

    train_eval_df, train_metrics, train_invest = print_result_block("Train", train_vis, selected_threshold)
    valid_eval_df, valid_metrics, valid_invest = print_result_block("Valid", valid_vis, selected_threshold)
    test_eval_df, test_metrics, test_invest = print_result_block("Test", test_vis, selected_threshold)

    # 그래프
    plt.figure(figsize=(12, 5))
    plt.plot(threshold_df["threshold"], threshold_df["f1"], label="Valid F1")
    plt.plot(threshold_df["threshold"], threshold_df["precision"], label="Valid Precision")
    plt.plot(threshold_df["threshold"], threshold_df["recall"], label="Valid Recall")
    plt.axvline(selected_threshold, linestyle="--", label=f"Selected={selected_threshold:.2f}")
    plt.title("Validation Classification Metrics by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(threshold_df["threshold"], threshold_df["avg_ann_return"], label="Valid Avg Ann Return")
    plt.plot(threshold_df["threshold"], threshold_df["win_rate_7pct"], label="Valid Win Rate >= 7%")
    plt.plot(threshold_df["threshold"], threshold_df["negative_rate"], label="Valid Negative Rate")
    plt.axvline(selected_threshold, linestyle="--", label=f"Selected={selected_threshold:.2f}")
    plt.title("Validation Investment Metrics by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(threshold_df["threshold"], threshold_df["trades_per_week"], label="Trades per Week")
    plt.axhline(TARGET_TRADES_PER_WEEK, linestyle="--", label="Target 1.0 / week")
    plt.axvline(selected_threshold, linestyle="--", label=f"Selected={selected_threshold:.2f}")
    plt.title("Validation Trades per Week by Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Trades per Week")
    plt.legend()
    plt.tight_layout()
    plt.show()

    buy_points = test_eval_df[test_eval_df["pred"] == 1]

    plt.figure(figsize=(14, 5))
    plt.plot(test_eval_df.index, test_eval_df["prob"], label="Predicted Probability")
    plt.scatter(buy_points.index, buy_points["prob"], marker="^", s=60, label="Selected Buy")
    plt.axhline(selected_threshold, linestyle="--", label=f"Threshold={selected_threshold:.2f}")
    plt.title("Test Probability with Weekly Top-1 Buy Signals")
    plt.xlabel("Date")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(14, 5))
    plt.plot(test_eval_df.index, test_eval_df["usd_krw"], label="USD/KRW")
    plt.scatter(buy_points.index, buy_points["usd_krw"], marker="^", s=60, label="Selected Buy")
    plt.title("USD/KRW with Selected Weekly Buy Signals (Test)")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    train_eval_df.to_csv(os.path.join(OUTPUT_DIR, "train_eval_selected_threshold.csv"), encoding="utf-8-sig")
    valid_eval_df.to_csv(os.path.join(OUTPUT_DIR, "valid_eval_selected_threshold.csv"), encoding="utf-8-sig")
    test_eval_df.to_csv(os.path.join(OUTPUT_DIR, "test_eval_selected_threshold.csv"), encoding="utf-8-sig")

    summary_df = pd.DataFrame([
        {"split": "train", **train_metrics, **train_invest, "selected_threshold": selected_threshold},
        {"split": "valid", **valid_metrics, **valid_invest, "selected_threshold": selected_threshold},
        {"split": "test", **test_metrics, **test_invest, "selected_threshold": selected_threshold},
    ])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_selected_threshold.csv"), index=False, encoding="utf-8-sig")

    print("\n저장 완료")
    print(os.path.join(OUTPUT_DIR, "threshold_search_valid_weekly.csv"))
    print(os.path.join(OUTPUT_DIR, "train_eval_selected_threshold.csv"))
    print(os.path.join(OUTPUT_DIR, "valid_eval_selected_threshold.csv"))
    print(os.path.join(OUTPUT_DIR, "test_eval_selected_threshold.csv"))
    print(os.path.join(OUTPUT_DIR, "summary_selected_threshold.csv"))


# ============================================================
# 13. 실행
# ============================================================

if __name__ == "__main__":
    # 1) 새로 학습하고 저장하려면
    train_and_save()

    # 2) 저장된 모델 기반으로 threshold / 투자 성과 평가하려면
    evaluate_saved_model()
