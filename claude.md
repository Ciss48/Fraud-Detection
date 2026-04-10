# Fraud Pattern Mining — Project Guide for Claude Code

## Project Overview

**Goal:** Khai thác các fraud pattern có thể giải thích được từ tập giao dịch thẻ tín dụng, sử dụng Association Rule Mining kết hợp với Anomaly Detection. Đây **không phải** bài toán dự đoán thuần túy — mục tiêu chính là khám phá ra các quy luật/pattern mô tả điều kiện nào khiến một giao dịch có khả năng là fraud cao, đủ để một fraud analyst đọc và hành động được.

**Dataset:** data/fraudTest.csv + fraudTrain.csv (you can combine them for analysis)

**Tech stack:** Python, Pandas, NumPy, mlxtend (Apriori/FP-Growth), scikit-learn (Isolation Forest), Matplotlib/Plotly, Jupyter Notebook

---

## Project Structure (3 Phases)

```
Anomaly Detection/
├── data/
│   ├── raw/              # fraudTrain.csv, fraudTest.csv gốc
│   └── processed/        # dữ liệu đã clean + feature engineering
├── notebooks/
│   ├── phase1_eda.ipynb
│   ├── phase2_pattern_mining.ipynb
│   └── phase3_detection_model.ipynb
├── src/
│   ├── preprocessing.py  # các hàm feature engineering tái sử dụng được
│   ├── rules.py          # các hàm liên quan association rule mining
│   └── evaluation.py     # precision, recall, overlap analysis
├── outputs/
│   ├── rules/            # rule tables output
│   └── figures/          # charts và visualizations
├── claude.md             # file này
└── requirements.txt
└── venv                  # enviroment
└── Fraud_Pattern_Mining_Proposal.docx
```

---

## Phase 1 — Exploratory Data Analysis (EDA)

**Mục tiêu:** Hiểu toàn bộ hình dạng của dữ liệu trước khi làm bất kỳ thứ gì. Không transform, không model — chỉ quan sát và note lại.

**Những gì cần khám phá:**

1. **Schema & data quality**
   - Shape, dtypes, missing values, duplicate rows
   - Phân bố `is_fraud` (class imbalance rate)

2. **Các cột numeric:** `amt`, `city_pop`, `lat`, `long`, `merch_lat`, `merch_long`
   - Distribution (histogram, boxplot)
   - So sánh fraud vs non-fraud theo từng cột

3. **Các cột categorical:** `category`, `gender`, `job`
   - Fraud rate theo từng giá trị
   - Top categories/jobs xuất hiện nhiều trong fraud

4. **Temporal features** (derive từ `trans_date_trans_time`)
   - Hour of day, day of week, month
   - Fraud rate theo từng time bucket

5. **Geographic distance** (derive từ lat/long vs merch_lat/merch_long)
   - Tính khoảng cách Haversine
   - So sánh distribution giữa fraud và non-fraud

6. **Cardholder age** (derive từ `dob`)
   - Fraud rate theo age group

7. **Correlations & anomalies nổi bật**
   - Có biến nào cực kỳ phân tách được fraud không?
   - Merchant nào có fraud rate cao bất thường?

**Output của Phase 1:**
- Notebook `phase1_eda.ipynb` với đầy đủ chart và observations
- Section **"EDA Notes"** bên dưới (điền sau khi chạy xong EDA)

---

### 📝 EDA Notes (điền sau khi hoàn thành Phase 1)

> **Điền vào đây sau khi đã chạy xong EDA. Những note này sẽ được dùng để điều chỉnh chiến lược cho Phase 2.**

#### Class imbalance
- Tỷ lệ fraud thực tế: `____%`
- Cần xử lý imbalance ở Phase 2/3 không? `[Yes / No]`

#### Biến quan trọng nhất (rank by discriminative power)
1. `___` — nhận xét:
2. `___` — nhận xét:
3. `___` — nhận xét:

#### Discretization bins — cần điều chỉnh không?
Proposal đề xuất các bins sau, có cần thay đổi dựa trên data thực không?

| Feature | Bins đề xuất trong proposal | Bins thực tế nên dùng | Lý do thay đổi (nếu có) |
|---|---|---|---|
| `amt` | low<$50, medium $50-200, high $200-500, very_high>$500 | | |
| `hour` | morning 6-12, afternoon 12-18, evening 18-22, night 22-6 | | |
| `distance` | local<10km, regional 10-100km, distant>100km | | |
| `city_pop` | rural<10k, suburban 10k-100k, urban>100k | | |
| `age` | young<30, middle 30-55, senior>55 | | |

#### Observations quan trọng cần nhớ cho Phase 2
- [ ] Category nào có fraud rate cao nhất:
- [ ] Time slot nào suspicious nhất:
- [ ] Distance pattern:
- [ ] Amount pattern:
- [ ] Bất thường khác:

#### Data quality issues cần xử lý
- [ ]
- [ ]

---

## Phase 2 — Fraud Pattern Mining

**Mục tiêu:** Khai thác các quy luật mô tả fraud pattern, so sánh với Isolation Forest baseline.

**Thực hiện sau khi hoàn thành EDA Notes ở trên.**

### 2.1 Feature Engineering & Discretization
- Parse `trans_date_trans_time` → `hour_of_day`, `day_of_week`, `month`
- Tính `distance_km` từ Haversine formula (lat/long vs merch_lat/merch_long)
- Tính `age` từ `dob` và transaction date
- Discretize các cột numeric theo bins đã quyết định ở EDA Notes
- Convert mỗi transaction thành transaction items cho rule mining

### 2.2 Baseline 1 — Simple Threshold Rule
- Flag tất cả transactions với `amt > $500` là fraud
- Tính Precision, Recall, F1

### 2.3 Baseline 2 — Isolation Forest
- Features: `amt`, `distance_km`, `city_pop`, `age`, `hour_of_day`
- Tuning: thử `contamination` ∈ {0.005, 0.01, 0.02}
- Tính Precision, Recall, F1 vs `is_fraud`

### 2.4 Main Method — Association Rule Mining
- Áp dụng FP-Growth (ưu tiên) hoặc Apriori trên fraud transaction set
- Threshold grid:
  - `min_support` ∈ {0.01%, 0.05%, 0.1%} (thấp vì fraud ~0.5%)
  - `min_confidence` ∈ {0.4, 0.6, 0.8}
  - `min_lift` ∈ {1.5, 2.0, 3.0}
- Chọn final threshold dựa trên: số rules hợp lý + interpretability + fraud recall

### 2.5 Comparison & Overlap Analysis
- Fraud bắt được bởi cả 2 phương pháp (strong overlap)
- Fraud bắt được bởi rules nhưng Isolation Forest miss (rule advantage)
- Fraud bắt được bởi Isolation Forest nhưng không có rule nào giải thích (anomaly advantage)

**Output của Phase 2:**
- Top fraud rules ranked by lift & confidence, với plain-language interpretation
- Rule summary table (support, confidence, lift, analyst action)
- Overlap analysis chart (Venn diagram)
- So sánh Precision/Recall/F1 của 3 phương pháp

---

## Phase 3 — Fraud Detection Model

**Mục tiêu:** Xây dựng mô hình phát hiện giao dịch đáng ngờ, kết hợp insights từ Phase 2.

### 3.1 Feature Set
- Dùng các features đã engineer ở Phase 2
- Thêm rule-based features: có/không match với top fraud rules
- Cân nhắc thêm features từ EDA Notes (những biến discriminative mạnh)

### 3.2 Xử lý Class Imbalance
- Thử SMOTE hoặc class weighting
- Quyết định dựa trên EDA Notes

### 3.3 Models
- **Primary:** Random Forest (interpretable feature importances)
- **Comparison:** XGBoost
- **Rule-augmented:** RF với thêm rule-based binary features từ Phase 2

### 3.4 Evaluation
- Metrics: Precision, Recall, F1, AUC-ROC, AUC-PR (dùng PR curve vì imbalanced)
- Cross-validation với stratified splits
- So sánh model thuần ML vs model có rule features

**Output của Phase 3:**
- Detection model với evaluation metrics
- Feature importance chart
- So sánh: rule-based (Phase 2) vs ML model vs hybrid

---

## Key Metrics Reference

| Metric | Dùng để đánh giá |
|---|---|
| Precision | Trong số các giao dịch bị flag là fraud, bao nhiêu là fraud thật |
| Recall | Trong số fraud thật, bao nhiêu được phát hiện |
| F1-Score | Harmonic mean của Precision và Recall |
| Support | Tỷ lệ transactions chứa itemset/antecedent |
| Confidence | P(fraud | antecedent conditions) |
| Lift | Fraud likelihood khi có rule conditions, so với base fraud rate. Lift > 1 = tốt |
| AUC-PR | Area under Precision-Recall curve, phù hợp hơn AUC-ROC cho imbalanced data |

---

## Notes for Claude Code

- Luôn kiểm tra shape và dtypes trước khi xử lý
- Tất cả notebooks phải reproducible (set random seed)
- Khi chạy FP-Growth trên 1.85M rows, cân nhắc sample stratified trước để test nhanh, sau đó mới chạy full
- Các hàm tái sử dụng (preprocessing, evaluation) nên được extract vào `src/`
- Mỗi phase output ra file riêng trong `outputs/` để có thể review độc lập
- Fraud rate rất thấp (~0.5%) — cẩn thận với accuracy paradox khi đánh giá model
