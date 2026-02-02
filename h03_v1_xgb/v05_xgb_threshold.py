import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils import resample
from xgboost import XGBClassifier

# 1. 데이터 로드
real_set = pd.read_csv("./_data/preprocessed_data.csv")
fake_set = pd.read_csv("./_data/augmented_dataset_10000_score_91_5.csv")

X = real_set.drop('Attrition', axis=1)
y = real_set['Attrition']

# 고정 설정
best_ratio = 0.65
thresholds = np.arange(0.3, 0.71, 0.02)
results = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"🚀 Threshold 최적화 실험 시작 (Mixing Ratio 1:{best_ratio})\n")

for thresh in thresholds:
    fold_f1 = []
    
    for train_index, val_index in skf.split(X, y):
        X_train_real, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_real, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 1:0.8 비율로 학습 데이터 구성
        train_0 = X_train_real[y_train_real == 0]
        train_1_real = X_train_real[y_train_real == 1]
        train_1_fake = fake_set[fake_set['Attrition'] == 1].drop('Attrition', axis=1)
        
        target_total_1 = int(len(train_0) * best_ratio)
        needed_n = target_total_1 - len(train_1_real)
        
        train_1_augmented = resample(train_1_fake, n_samples=needed_n, replace=False, random_state=42)
        X_train_combined = pd.concat([train_0, train_1_real, train_1_augmented])
        y_train_combined = [0] * len(train_0) + [1] * (len(train_1_real) + len(train_1_augmented))
        
        # 모델 학습
        model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
        model.fit(X_train_combined, y_train_combined)
        
        # 확률값 예측 (클래스 1일 확률 추출)
        probs = model.predict_proba(X_val)[:, 1]
        
        # 설정한 임계값 적용
        preds = (probs >= thresh).astype(int)
        fold_f1.append(f1_score(y_val, preds))
        
    avg_score = np.mean(fold_f1)
    results.append(avg_score)
    print(f"🔹 Threshold {thresh:.2f} -> Avg F1-Score: {avg_score:.4f}")

# 2. 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(thresholds, results, marker='s', color='darkorange', linewidth=2)
plt.title(f'Threshold vs F1-Score (Ratio 1:{best_ratio})', fontsize=14)
plt.xlabel('Classification Threshold', fontsize=12)
plt.ylabel('Average F1-Score', fontsize=12)
plt.axvline(thresholds[np.argmax(results)], color='red', linestyle='--', label='Best Threshold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

print(f"\n✅ 최적 임계값: {thresholds[np.argmax(results)]:.2f}")
print(f"✅ 최고 F1-Score: {max(results):.4f}")

# 🚀 Threshold 최적화 실험 시작 (Mixing Ratio 1:0.8)

# 🔹 Threshold 0.30 -> Avg F1-Score: 0.4746
# 🔹 Threshold 0.32 -> Avg F1-Score: 0.4802
# 🔹 Threshold 0.34 -> Avg F1-Score: 0.4775
# 🔹 Threshold 0.36 -> Avg F1-Score: 0.4911
# 🔹 Threshold 0.38 -> Avg F1-Score: 0.5003
# 🔹 Threshold 0.40 -> Avg F1-Score: 0.4960
# 🔹 Threshold 0.42 -> Avg F1-Score: 0.5041
# 🔹 Threshold 0.44 -> Avg F1-Score: 0.5103
# 🔹 Threshold 0.46 -> Avg F1-Score: 0.5163
# 🔹 Threshold 0.48 -> Avg F1-Score: 0.5178
# 🔹 Threshold 0.50 -> Avg F1-Score: 0.5195
# 🔹 Threshold 0.52 -> Avg F1-Score: 0.5281
# 🔹 Threshold 0.54 -> Avg F1-Score: 0.5417
# 🔹 Threshold 0.56 -> Avg F1-Score: 0.5428
# 🔹 Threshold 0.58 -> Avg F1-Score: 0.5404
# 🔹 Threshold 0.60 -> Avg F1-Score: 0.5387
# 🔹 Threshold 0.62 -> Avg F1-Score: 0.5451
# 🔹 Threshold 0.64 -> Avg F1-Score: 0.5490
# 🔹 Threshold 0.66 -> Avg F1-Score: 0.5451
# 🔹 Threshold 0.68 -> Avg F1-Score: 0.5172
# 🔹 Threshold 0.70 -> Avg F1-Score: 0.5031

# ✅ 최적 임계값: 0.64
# ✅ 최고 F1-Score: 0.5490