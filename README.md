# 📈 USD/KRW 환율 기반 장기 투자 의사결정 모델 (Transformer)

---

## 1. 프로젝트 개요

본 프로젝트는 **원/달러 환율과 거시경제 변수 간의 관계를 기반으로,
달러 투자 여부를 판단하는 머신러닝 모델을 구축하는 것**을 목표로 한다.

단순 환율 예측이 아니라,

> “지금 달러를 매수하면 6개월 이상 보유했을 때 유의미한 수익을 얻을 수 있는가?”

를 판단하는 **장기 투자 의사결정 모델**을 설계하였다.

---

## 2. 문제 정의

### 🎯 목표

* 달러 투자 매수 여부 판단 (Binary Classification)

### 📌 투자 전략

* 최소 보유 기간: **6개월 이상**
* 목표 수익률: **연 7% 이상**

### 🧮 타깃 정의

```python
ann_return = ((future_price / current_price) - 1) ** (365 / HOLD_DAYS) - 1
target = 1 if ann_return >= 0.07 else 0
```

---

## 3. 데이터 구성

### 📊 (1) 금융시장 데이터 (yfinance)

* USD/KRW 환율 (`KRW=X`)
* 달러 인덱스 (`DX-Y.NYB`)
* S&P500 (`^GSPC`)
* KOSPI (`^KS11`)
* VIX (`^VIX`)
* 원자재 (금, 유가 등)

👉 글로벌 자본 흐름 및 위험 선호 반영

---

### 🏦 (2) 거시경제 데이터 (ECOS)

* 경상수지
* CD금리 (91일)
* (M2는 총량 데이터 확보 후 추가 예정)

👉 한국 경제 펀더멘탈 반영

---

## 4. 데이터 전처리

* 시계열 기준 정렬
* 월별 데이터 → 일별 forward fill
* 결측치 처리 (ffill / bfill)
* 로그 변환 및 정규화

---

## 5. Feature Engineering

논문 기반 변수 설계를 반영하여 생성

### 주요 변수

* 로그 변환

  * `ln_usd_krw`, `ln_kospi`, `ln_sp500`, `ln_dxy`

* 수익률

  * 1일 / 5일 / 20일 변화율

* 이동평균

  * MA(5, 10, 20, 60)

* 변동성

  * rolling std

* 기술지표

  * RSI
  * MACD
  * Bollinger Bands

* 거시 변수

  * 금리 수준
  * 경상수지 변화

---

## 6. 시계열 데이터 구성

* 입력 길이: `180일 (약 6개월)`
* 출력: 현재 시점의 투자 판단

```python
X.shape = (batch, sequence_length, num_features)
```

---

## 7. 모델 구조 (Transformer)

### 🧠 Architecture

* Input Projection Layer
* Positional Encoding
* Transformer Encoder (Multi-head Attention)
* Feed Forward Network
* Classification Head (Sigmoid)

### ⚙️ 주요 파라미터

```python
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
```

---

## 8. 학습 전략

* Loss: Binary Cross Entropy
* Optimizer: Adam
* Walk-forward validation

```text
Train: 과거 데이터
Test: 미래 데이터
```

👉 시계열 데이터 누수 방지

---

## 9. 평가 방식

### 📊 모델 성능

* ROC-AUC
* Accuracy
* Precision / Recall

### 💰 투자 성과 (Backtest)

* CAGR
* 누적 수익률
* MDD (최대 낙폭)
* Sharpe Ratio
* 매수 신호 정확도

---

## 10. 투자 의사결정 로직

```python
if prob >= 0.65:
    BUY
else:
    HOLD
```

* 매수 후 최소 6개월 보유
* 이후 재평가

---

## 11. 프로젝트 구조

```
project/
│
├── data/
│   ├── raw_data.csv
│   ├── processed_data.csv
│
├── notebook/
│   └── usd_krw_transformer.ipynb
│
├── model/
│   └── transformer.py
│
├── utils/
│   └── data_loader.py
│
└── README.md
```

---

## 12. 한계 및 개선 방향

### ⚠️ 한계

* ECOS 일부 데이터 수집 실패 가능성
* 월간 데이터의 발표 시차 미반영
* Transformer의 데이터 요구량 문제

### 🚀 개선 방향

* 미국 거시경제 변수 추가
* Attention 기반 해석 가능성 분석
* Ensemble (LGBM + Transformer)
* M2 총량 데이터 추가
* 강화학습 기반 전략 확장

---

## 13. 핵심 인사이트

* 환율은 단순 시계열이 아니라
  **거시경제 + 글로벌 자산 흐름의 결과**
* Transformer는
  **장기 패턴 학습에 유리**
* 단순 예측보다
  **의사결정 기준 모델이 더 실용적**

---

## 14. 결론

본 프로젝트는

> 환율을 예측하는 모델이 아니라,
> **달러 투자 의사결정을 자동화하는 시스템**

을 목표로 한다.

이를 통해 장기 투자 관점에서
보다 안정적인 전략 수립이 가능하다.

---

## 👨‍💻 Author

* 데이터 기반 의사결정 및 시계열 모델링에 관심
* 거시경제 + 머신러닝 결합 모델 연구

---
