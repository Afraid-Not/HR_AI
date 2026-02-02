import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils import resample
from xgboost import XGBClassifier

# 1. 데이터 로드
real_set = pd.read_csv("./_data/preprocessed_data.csv")
fake_set = pd.read_csv("./_data/augmented_dataset_10000_score_91_5.csv")
# 1. 데이터 준비
X = real_set.drop('Attrition', axis=1)
y = real_set['Attrition']

# 스캔 범위 설정 (0.01 단위)
ratios = np.arange(0.3, 0.81, 0.01)
thresholds = np.arange(0.2, 0.81, 0.01)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_results_per_ratio = []

print(f"🚀 [2D Grid Search] 비율 51개 x 임계값 61개 탐색 시작...")

for ratio in ratios:
    # 각 폴드별 확률값을 저장할 리스트
    all_y_val = []
    all_probs = []
    
    for train_index, val_index in skf.split(X, y):
        X_train_real, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_real, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 데이터 증강 (Ratio 적용)
        train_0 = X_train_real[y_train_real == 0]
        train_1_real = X_train_real[y_train_real == 1]
        train_1_fake = fake_set[fake_set['Attrition'] == 1].drop('Attrition', axis=1)
        
        target_total_1 = int(len(train_0) * ratio)
        needed_n = max(0, target_total_1 - len(train_1_real))
        
        if needed_n > 0:
            train_1_aug = resample(train_1_fake, n_samples=needed_n, replace=False, random_state=42)
            X_train_comb = pd.concat([train_0, train_1_real, train_1_aug])
            y_train_comb = [0]*len(train_0) + [1]*(len(train_1_real) + len(train_1_aug))
        else:
            X_train_comb = pd.concat([train_0, train_1_real])
            y_train_comb = [0]*len(train_0) + [1]*len(train_1_real)
            
        # 모델 학습 (XGBoost)
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss')
        model.fit(X_train_comb, y_train_comb)
        
        # 예측 확률 저장
        probs = model.predict_proba(X_val)[:, 1]
        all_probs.extend(probs)
        all_y_val.extend(y_val)
    
    # 해당 Ratio에서 최적의 Threshold 찾기
    best_f1_for_ratio = -1
    best_thresh_for_ratio = -1
    
    for thresh in thresholds:
        preds = (np.array(all_probs) >= thresh).astype(int)
        score = f1_score(all_y_val, preds)
        if score > best_f1_for_ratio:
            best_f1_for_ratio = score
            best_thresh_for_ratio = thresh
            
    best_results_per_ratio.append({
        'Ratio': round(ratio, 2),
        'Best_Threshold': round(best_thresh_for_ratio, 2),
        'Max_F1': round(best_f1_for_ratio, 4)
    })
    
    # 0.1 단위로 진행 상황 출력
    if round(ratio, 2) % 0.1 == 0:
        print(f"📍 Ratio {ratio:.2f} 완료 (Best Thresh: {best_thresh_for_ratio:.2f}, F1: {best_f1_for_ratio:.4f})")

# 최종 결과 데이터프레임
df_res = pd.DataFrame(best_results_per_ratio)
print("\n🏆 [최종 스캔 결과 상위 10개]")
print(df_res.sort_values(by='Max_F1', ascending=False).head(10))


# 🚀 [2D Grid Search] 비율 51개 x 임계값 61개 탐색 시작...
# 📍 Ratio 0.40 완료 (Best Thresh: 0.34, F1: 0.5236)
# 📍 Ratio 0.80 완료 (Best Thresh: 0.60, F1: 0.5553)

# 🏆 [최종 스캔 결과 상위 10개]
#     Ratio  Best_Threshold  Max_F1
# 50   0.80            0.60  0.5553
# 37   0.67            0.48  0.5492
# 1    0.31            0.26  0.5445
# 49   0.79            0.54  0.5444
# 0    0.30            0.38  0.5439
# 48   0.78            0.59  0.5428
# 8    0.38            0.32  0.5425
# 41   0.71            0.53  0.5423
# 31   0.61            0.47  0.5414
# 47   0.77            0.52  0.5402