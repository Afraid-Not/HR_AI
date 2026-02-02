import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils import resample
from xgboost import XGBClassifier
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# 1. 데이터 로드
# 재현님의 데이터 경로에 맞춰 로드합니다.
real_set = pd.read_csv("./_data/preprocessed_data.csv")
fake_set = pd.read_csv("./_data/augmented_dataset_10000_score_91_5.csv")

X = real_set.drop('Attrition', axis=1)
y = real_set['Attrition']

# 2. 스캔 범위 설정 (0.01 단위로 정밀 탐색)
ratios = np.arange(0.3, 0.81, 0.01)       # 51개 구간
thresholds = np.arange(0.2, 0.81, 0.01)   # 61개 구간

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 전체 그리드 결과를 저장할 리스트
grid_scores = []
# Ratio별 최고 성적을 저장할 리스트
best_results_per_ratio = []

print(f"🚀 [2D Grid Search] 탐색 시작: 비율 {len(ratios)}개 x 임계값 {len(thresholds)}개")

# 3. 메인 루프: 비율(Ratio) 탐색
for ratio in ratios:
    all_y_val = []
    all_probs = []
    
    # 5-Fold 교차 검증 수행
    for train_index, val_index in skf.split(X, y):
        X_train_real, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_real, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 데이터 증강 로직 (Ratio 적용)
        train_0 = X_train_real[y_train_real == 0]
        train_1_real = X_train_real[y_train_real == 1]
        train_1_fake = fake_set[fake_set['Attrition'] == 1].drop('Attrition', axis=1)
        
        target_total_1 = int(len(train_0) * ratio)
        needed_n = max(0, target_total_1 - len(train_1_real))
        
        if needed_n > 0:
            # 부족한 만큼 가짜 데이터에서 샘플링
            train_1_aug = resample(train_1_fake, n_samples=needed_n, replace=False, random_state=42)
            X_train_comb = pd.concat([train_0, train_1_real, train_1_aug])
            y_train_comb = [0]*len(train_0) + [1]*(len(train_1_real) + len(train_1_aug))
        else:
            X_train_comb = pd.concat([train_0, train_1_real])
            y_train_comb = [0]*len(train_0) + [1]*len(train_1_real)
            
        # 모델 학습 (XGBoost)
        model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            random_state=42, 
            eval_metric='logloss'
        )
        model.fit(X_train_comb, y_train_comb)
        
        # 예측 확률 저장 (OOF 방식)
        probs = model.predict_proba(X_val)[:, 1]
        all_probs.extend(probs)
        all_y_val.extend(y_val)
    
    # 해당 Ratio에서 모든 Threshold에 대해 점수 계산
    ratio_max_f1 = -1
    ratio_best_thresh = -1
    
    for thresh in thresholds:
        preds = (np.array(all_probs) >= thresh).astype(int)
        score = f1_score(all_y_val, preds)
        
        # 1) 히트맵용 데이터 저장
        grid_scores.append({
            'Ratio': round(ratio, 2),
            'Threshold': round(thresh, 2),
            'F1_Score': score
        })
        
        # 2) Ratio별 최고점 업데이트
        if score > ratio_max_f1:
            ratio_max_f1 = score
            ratio_best_thresh = thresh
            
    best_results_per_ratio.append({
        'Ratio': round(ratio, 2),
        'Best_Threshold': round(ratio_best_thresh, 2),
        'Max_F1': round(ratio_max_f1, 4)
    })
    
    # 진행 상황 출력
    if round(ratio, 2) % 0.1 == 0:
        print(f"📍 진행 중... Ratio {ratio:.2f} 완료 (Best Thresh: {ratio_best_thresh:.2f}, F1: {ratio_max_f1:.4f})")

# 4. 결과 분석 및 시각화
df_heatmap = pd.DataFrame(grid_scores)
pivot_table = df_heatmap.pivot(index="Ratio", columns="Threshold", values="F1_Score")

# 히트맵 그리기
plt.figure(figsize=(16, 10))
sns.heatmap(pivot_table, annot=False, cmap="YlGnBu", cbar_kws={'label': 'F1-Score'})

# 최고점 좌표 찾기
best_idx = df_heatmap['F1_Score'].idxmax()
best_row = df_heatmap.loc[best_idx]

plt.title(f"Attrition Prediction F1-Score Heatmap\n(Global Max: {best_row['F1_Score']:.4f} at Ratio {best_row['Ratio']}, Threshold {best_row['Threshold']})", fontsize=15)
plt.xlabel("Classification Threshold", fontsize=12)
plt.ylabel("Synthetic Data Ratio (1:X)", fontsize=12)

# 이미지 저장
plt.tight_layout()
plt.savefig("./_data/f1_score_optimization_heatmap.png", dpi=300)
print(f"\n✅ 히트맵 시각화 완료 및 './_data/f1_score_optimization_heatmap.png' 저장 성공")

# 5. 최종 상위 결과 출력
print("\n🏆 [최종 스캔 결과 상위 10개 조합]")
df_res = pd.DataFrame(best_results_per_ratio)
print(df_res.sort_values(by='Max_F1', ascending=False).head(10))

plt.show()


# 🚀 [2D Grid Search] 탐색 시작: 비율 51개 x 임계값 62개
# 📍 진행 중... Ratio 0.40 완료 (Best Thresh: 0.34, F1: 0.5236)
# 📍 진행 중... Ratio 0.80 완료 (Best Thresh: 0.60, F1: 0.5553)

# ✅ 히트맵 시각화 완료 및 './_data/f1_score_optimization_heatmap.png' 저장 성공

# 🏆 [최종 스캔 결과 상위 10개 조합]
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