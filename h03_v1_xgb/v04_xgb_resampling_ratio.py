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

# 테스트할 비율 리스트 (1:0.2부터 1:1.0까지)
ratios = np.arange(0.2, 0.99, 0.03)
results = []

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("🚀 비율별 성능 최적화 실험 시작\n")

for ratio in ratios:
    fold_f1 = []
    
    for train_index, val_index in skf.split(X, y):
        # 폴드 분리
        X_train_real, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_real, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 실제 데이터 클래스 분리
        train_0 = X_train_real[y_train_real == 0]
        train_1_real = X_train_real[y_train_real == 1]
        
        # 가짜 데이터에서 퇴사자만 분리
        train_1_fake = fake_set[fake_set['Attrition'] == 1].drop('Attrition', axis=1)
        
        # [핵심] 목표 퇴사자 수 계산 및 부족분 채우기
        target_total_1 = int(len(train_0) * ratio)
        needed_n = target_total_1 - len(train_1_real)
        
        if needed_n > 0:
            # 부족한 만큼 가짜 데이터에서 샘플링 (변수명 통일: needed_n)
            train_1_augmented = resample(train_1_fake, n_samples=needed_n, replace=False, random_state=42)
            X_train_combined = pd.concat([train_0, train_1_real, train_1_augmented])
            # y값 생성: 재직자(0) 수만큼 0, 퇴사자(1) 총합만큼 1
            y_train_combined = [0] * len(train_0) + [1] * (len(train_1_real) + len(train_1_augmented))
        else:
            X_train_combined = pd.concat([train_0, train_1_real])
            y_train_combined = [0] * len(train_0) + [1] * len(train_1_real)
            
        # 모델 학습 (XGBoost)
        model = XGBClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=42, 
            eval_metric='logloss'
        )
        model.fit(X_train_combined, y_train_combined)
        
        # 평가
        preds = model.predict(X_val)
        fold_f1.append(f1_score(y_val, preds))
        
    avg_score = np.mean(fold_f1)
    results.append(avg_score)
    print(f"✅ Ratio 1:{ratio:.1f} -> Avg F1-Score: {avg_score:.4f}")

# 2. 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(ratios, results, marker='o', linestyle='-', color='b', linewidth=2)
plt.title('Mixing Ratio vs Attrition F1-Score (5-Fold Avg)', fontsize=14)
plt.xlabel('Synthetic Attrition Ratio (1:X)', fontsize=12)
plt.ylabel('Average F1-Score', fontsize=12)
plt.xticks(ratios)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
#  비율별 성능 최적화 실험 시작

# ✅ Ratio 1:0.2 -> Avg F1-Score: 0.4394
# ✅ Ratio 1:0.2 -> Avg F1-Score: 0.4618
# ✅ Ratio 1:0.3 -> Avg F1-Score: 0.4855
# ✅ Ratio 1:0.3 -> Avg F1-Score: 0.5005
# ✅ Ratio 1:0.3 -> Avg F1-Score: 0.4936
# ✅ Ratio 1:0.3 -> Avg F1-Score: 0.4909
# ✅ Ratio 1:0.4 -> Avg F1-Score: 0.4936
# ✅ Ratio 1:0.4 -> Avg F1-Score: 0.5066
# ✅ Ratio 1:0.4 -> Avg F1-Score: 0.4972
# ✅ Ratio 1:0.5 -> Avg F1-Score: 0.5080
# ✅ Ratio 1:0.5 -> Avg F1-Score: 0.4960
# ✅ Ratio 1:0.5 -> Avg F1-Score: 0.5034
# ✅ Ratio 1:0.6 -> Avg F1-Score: 0.5213
# ✅ Ratio 1:0.6 -> Avg F1-Score: 0.5299
# ✅ Ratio 1:0.6 -> Avg F1-Score: 0.5040
# ✅ Ratio 1:0.6 -> Avg F1-Score: 0.5375
# ✅ Ratio 1:0.7 -> Avg F1-Score: 0.5239
# ✅ Ratio 1:0.7 -> Avg F1-Score: 0.5335
# ✅ Ratio 1:0.7 -> Avg F1-Score: 0.5260
# ✅ Ratio 1:0.8 -> Avg F1-Score: 0.5373
# ✅ Ratio 1:0.8 -> Avg F1-Score: 0.5195
# ✅ Ratio 1:0.8 -> Avg F1-Score: 0.5365
# ✅ Ratio 1:0.9 -> Avg F1-Score: 0.5183
# ✅ Ratio 1:0.9 -> Avg F1-Score: 0.5275
# ✅ Ratio 1:0.9 -> Avg F1-Score: 0.5196
# ✅ Ratio 1:0.9 -> Avg F1-Score: 0.5111
# ✅ Ratio 1:1.0 -> Avg F1-Score: 0.5183