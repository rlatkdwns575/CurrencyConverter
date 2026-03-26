# 📈 USD/KRW 환율 기반 장기 투자 의사결정 시스템 (Transformer)

---

## 1. 프로젝트 개요

본 프로젝트는 **원/달러 환율과 거시경제 및 글로벌 금융 데이터를 기반으로,
달러 투자 여부를 판단하는 머신러닝 시스템을 구축하는 것**을 목표로 한다.

단순히 환율을 예측하는 것이 아니라,

> “현재 시점에서 달러를 매수하면,
> 6개월 이상 보유했을 때 유의미한 수익을 얻을 수 있는가?”

를 판단하는 **장기 투자 의사결정 모델**을 설계하였다.

---

## 2. 문제 정의

### 🎯 목표

* 달러 매수 여부 판단 (Binary Classification)

### 📌 투자 전략

* 최소 보유 기간: **6개월 이상**
* 목표 수익률: **연 7% 이상**

### 🧮 타깃 정의

```python
ann_return = ((future_price / current_price) - 1) ** (365 / HOLD_DAYS) - 1
target = 1 if ann_return >= 0.07 else 0
```

👉 단순 상승 여부가 아닌
**투자 관점의 수익률 기반 라벨링**

---

## 3. 데이터 구성

### 📊 (1) 금융시장 데이터 (yfinance)

* USD/KRW 환율 (`KRW=X`)
* 달러 인덱스 (`DX-Y.NYB`)
* S&P500 (`^GSPC`)
* KOSPI (`^KS11`)
* VIX (`^VIX`)
* 원자재 (금, 유가 등)

👉 글로벌 자본 흐름 및 리스크 선호 반영

---

### 🏦 (2) 거시경제 데이터 (ECOS)

* 경상수지 (월별)
* CD금리 (91일)
* (M2는 총량 데이터 확보 후 추가 예정)

👉 한국 경제 펀더멘탈 반영

---

## 4. 데이터 전처리

* 시계열 기준 정렬
* 월별 데이터 → 일별 forward fill
* 결측치 처리 (ffill / bfill)
* 이상치 제거
* 로그 변환 및 정규화

---

## 5. Feature Engineering

논문 기반 변수 설계를 반영하여 환율 설명력을 강화하였다.

### 주요 변수

#### 1) 로그 변환

* `ln_usd_krw`
* `ln_kospi`
* `ln_sp500`
* `ln_dxy`

#### 2) 수익률

* 1일 / 5일 / 20일 변화율

#### 3) 이동평균

* MA(5, 10, 20, 60)

#### 4) 변동성

* rolling std (10, 20)

#### 5) 기술지표

* RSI (14)
* MACD
* Bollinger Bands

#### 6) 거시 변수

* 금리 수준
* 경상수지 변화율

---

## 6. 시계열 데이터 구성

* 입력 길이: **180일 (약 6개월)**
* 출력: 현재 시점의 투자 판단

```python
X.shape = (batch, sequence_length, num_features)
```

👉 과거 흐름을 기반으로 미래 투자 여부 판단

---

## 7. 모델 구조

---

### 🧠 7.1 Transformer 모델

#### 구조

* Input Projection
* Positional Encoding
* Multi-Head Attention
* Feed Forward Network
* Classification Head (Sigmoid)

#### 주요 파라미터

```python
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
```

#### 특징

* 장기 시계열 의존성 학습
* 변수 간 상호작용 반영
* Attention 기반 중요 구간 학습

---

### 🔁 7.2 Baseline 모델 (LSTM)

#### 구조

* LSTM Layer
* Fully Connected Layer
* Sigmoid Output

#### 목적

* Transformer와 성능 비교
* 복잡한 모델 사용의 타당성 검증

---

## 8. 학습 전략

* Loss: Binary Cross Entropy
* Optimizer: Adam
* Early Stopping 적용

### 📌 데이터 분할

| 구분         | 기간     |
| ---------- | ------ |
| Train      | 과거 데이터 |
| Validation | 중간 구간  |
| Test       | 최신 데이터 |

👉 시계열 순서를 유지한 분할

---

### 📌 검증 방식

* Walk-forward validation
* 연도별 성능 평가

👉 데이터 누수 방지

---

## 9. 성능 평가

---

### 📊 모델 성능 지표

* Accuracy
* ROC-AUC
* Precision / Recall
* F1-score

---

### 💰 투자 성과 지표

* CAGR (연평균 수익률)
* 누적 수익률
* MDD (최대 낙폭)
* Sharpe Ratio
* 매수 성공률

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

👉 모델 → 실제 투자 전략 연결

---

## 11. 결과 시각화

### 📊 표

| 모델          | Accuracy | ROC-AUC | CAGR | MDD | Sharpe |
| ----------- | -------- | ------- | ---- | --- | ------ |
| LSTM        | -        | -       | -    | -   | -      |
| Transformer | -        | -       | -    | -   | -      |

---

### 📈 그래프

* 학습 Loss 곡선
* Validation 성능 비교
* 실제 환율 vs 예측 확률
* 누적 수익률 곡선
* 매수 시점 표시
* 모델별 백테스트 비교

---

## 12. 시스템 확장 (GUI)

### 🔧 구현 방식

* **Streamlit 기반 웹 UI**

### 주요 기능

* 데이터 수집 실행
* 모델 선택 (LSTM / Transformer)
* 학습 실행
* 최신 데이터 기반 투자 판단
* 매수 확률 출력
* 백테스트 결과 시각화

👉 버튼 클릭으로 전체 파이프라인 실행 가능

---

## 13. 프로젝트 구조

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
│   ├── transformer.py
│   ├── lstm.py
│
├── utils/
│   ├── data_loader.py
│   ├── feature_engineering.py
│
├── app/
│   └── streamlit_app.py
│
└── README.md
```

---

## 14. 한계 및 개선 방향

### ⚠️ 한계

* ECOS 데이터 일부 수집 불안정
* 월간 데이터 발표 시차 미반영
* Transformer 학습 데이터 부족 가능성

---

### 🚀 개선 방향

* 미국 거시경제 변수 추가
* Attention 해석 모델 적용
* Ensemble (LGBM + Transformer)
* 강화학습 기반 투자 전략 확장
* 실시간 데이터 파이프라인 구축

---

## 15. 핵심 인사이트

* 환율은 단순 시계열이 아니라
  **거시경제 + 글로벌 자산 흐름의 결과**
* Transformer는 장기 패턴 학습에 유리
* 예측보다 **의사결정 모델이 더 실용적**

---

## 16. 결론

본 프로젝트는

> 환율을 예측하는 모델이 아니라
> **달러 투자 의사결정을 자동화하는 시스템**

을 목표로 한다.

이를 통해 장기 투자 관점에서
보다 안정적인 투자 전략 수립이 가능하다.

---

## 👨‍💻 Author

* Data-driven 의사결정 및 시계열 모델링 연구
* 금융 + 머신러닝 융합 프로젝트 수행

---
