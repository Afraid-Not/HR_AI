import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os  # 디렉토리 생성을 위해 추가
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils import resample
from lightgbm import LGBMClassifier
import warnings

# 경고 메시지 무시 (LGBM 특유의 정보성 로그 포함)
warnings.filterwarnings('ignore')

# 1. 데이터 로드
real_set = pd.read_csv("./_data/preprocessed_data.csv")
fake_set = pd.read_csv("./_data/augmented_dataset_10000_score_91_5.csv")

X = real_set.drop('Attrition', axis=1)
y = real_set['Attrition']

# 2. 결과 저장 경로 설정 및 생성
save_dir = "./h04_v2_lgbm"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"📂 {save_dir} 폴더가 생성되었습니다.")

# 3. 스캔 범위 설정
ratios = np.arange(0.3, 0.81, 0.01)
thresholds = np.arange(0.2, 0.81, 0.01)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_scores = []
best_results_per_ratio = []

print(f"🚀 [2D Grid Search] LGBM 탐색 시작: 비율 {len(ratios)}개 x 임계값 {len(thresholds)}개")

# 4. 메인 루프: 비율(Ratio) 탐색
for ratio in ratios:
    all_y_val = []
    all_probs = []
    
    for train_index, val_index in skf.split(X, y):
        X_train_real, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_real, y_val = y.iloc[train_index], y.iloc[val_index]
        
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
            
        # 모델 학습 (LightGBM)
        # LGBM은 verbosity=-1로 설정해야 불필요한 로그를 줄일 수 있습니다.
        model = LGBMClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5,
            num_leaves=31, # max_depth 5에 적합한 설정
            random_state=42,
            verbosity=-1, # 로그 최소화
            importance_type='gain' # 피처 중요도 기준 설정
        )
        # eval_metric은 fit 함수에서 전달하는 것이 권장됩니다.
        model.fit(X_train_comb, y_train_comb, eval_metric='logloss')
        
        probs = model.predict_proba(X_val)[:, 1]
        all_probs.extend(probs)
        all_y_val.extend(y_val)
    
    ratio_max_f1 = -1
    ratio_best_thresh = -1
    
    for thresh in thresholds:
        preds = (np.array(all_probs) >= thresh).astype(int)
        score = f1_score(all_y_val, preds)
        
        grid_scores.append({
            'Ratio': round(ratio, 2),
            'Threshold': round(thresh, 2),
            'F1_Score': score
        })
        
        if score > ratio_max_f1:
            ratio_max_f1 = score
            ratio_best_thresh = thresh
            
    best_results_per_ratio.append({
        'Ratio': round(ratio, 2),
        'Best_Threshold': round(ratio_best_thresh, 2),
        'Max_F1': round(ratio_max_f1, 4)
    })
    
    if round(ratio, 2) % 0.1 == 0:
        print(f"📍 Ratio {ratio:.2f} 완료 (Best Thresh: {ratio_best_thresh:.2f}, F1: {ratio_max_f1:.4f})")

# 5. 결과 분석 및 시각화
df_heatmap = pd.DataFrame(grid_scores)
pivot_table = df_heatmap.pivot(index="Ratio", columns="Threshold", values="F1_Score")

plt.figure(figsize=(16, 10))
sns.heatmap(pivot_table, annot=False, cmap="YlGnBu", cbar_kws={'label': 'F1-Score'})

best_idx = df_heatmap['F1_Score'].idxmax()
best_row = df_heatmap.loc[best_idx]

plt.title(f"LGBM Attrition F1-Score Heatmap\n(Max: {best_row['F1_Score']:.4f} at R:{best_row['Ratio']}, T:{best_row['Threshold']})", fontsize=15)
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("Ratio", fontsize=12)

# 이미지 저장
plt.tight_layout()
result_path = os.path.join(save_dir, "v01_result.png")
plt.savefig(result_path, dpi=300)
print(f"\n✅ 히트맵 시각화 완료 및 '{result_path}' 저장 성공")

# 6. 최종 상위 결과 출력
print("\n🏆 [최종 스캔 결과 상위 10개 조합]")
df_res = pd.DataFrame(best_results_per_ratio)
print(df_res.sort_values(by='Max_F1', ascending=False).head(10))

plt.show()

#  [2D Grid Search] LGBM 탐색 시작: 비율 51개 x 임계값 62개
# 📍 Ratio 0.40 완료 (Best Thresh: 0.40, F1: 0.5168)
# 📍 Ratio 0.80 완료 (Best Thresh: 0.61, F1: 0.5485)

# ✅ 히트맵 시각화 완료 및 './h04_v2_lgbm\v01_result.png' 저장 성공

# 🏆 [최종 스캔 결과 상위 10개 조합]
#     Ratio  Best_Threshold  Max_F1
# 38   0.68            0.51  0.5573
# 49   0.79            0.63  0.5572
# 46   0.76            0.51  0.5543
# 43   0.73            0.52  0.5497
# 50   0.80            0.61  0.5485
# 24   0.54            0.42  0.5471
# 28   0.58            0.52  0.5458
# 48   0.78            0.55  0.5447
# 26   0.56            0.43  0.5428
# 22   0.52            0.38  0.5421