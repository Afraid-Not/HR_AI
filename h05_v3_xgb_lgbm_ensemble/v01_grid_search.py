import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils import resample
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')

# 1. 데이터 로드
real_set = pd.read_csv("./_data/preprocessed_data.csv")
fake_set = pd.read_csv("./_data/augmented_dataset_10000_score_91_5.csv")
X = real_set.drop('Attrition', axis=1)
y = real_set['Attrition']

save_dir = "./h05_v3_xgb_lgbm_ensemble"
if not os.path.exists(save_dir): os.makedirs(save_dir)

# 2. 탐색 범위 설정
ratios = np.arange(0.3, 0.81, 0.02)      # 속도를 위해 0.02 단위 (조절 가능)
weights = np.arange(0.1, 1.0, 0.1)      # XGB 가중치 (0.1 ~ 0.9)
thresholds = np.arange(0.2, 0.81, 0.01)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_overall = {'f1': -1}
all_results = []

print(f"🚀 [3D Grid Search] Ratio x Weight x Threshold 탐색 시작...")

# 3. 메인 루프 (Ratio)
for ratio in ratios:
    oof_xgb_probs = np.zeros(len(y))
    oof_lgbm_probs = np.zeros(len(y))
    
    for train_index, val_index in skf.split(X, y):
        X_train_real, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_real, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 증강 로직
        train_0 = X_train_real[y_train_real == 0]
        train_1_real = X_train_real[y_train_real == 1]
        train_1_fake = fake_set[fake_set['Attrition'] == 1].drop('Attrition', axis=1)
        target_n = int(len(train_0) * ratio)
        needed_n = max(0, target_n - len(train_1_real))
        
        train_1_aug = resample(train_1_fake, n_samples=needed_n, replace=False, random_state=42)
        X_train_comb = pd.concat([train_0, train_1_real, train_1_aug])
        y_train_comb = [0]*len(train_0) + [1]*(len(train_1_real) + len(train_1_aug))
        
        # 모델 학습 (XGB, LGBM)
        m_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, eval_metric='logloss')
        m_lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=-1)
        
        m_xgb.fit(X_train_comb, y_train_comb)
        m_lgbm.fit(X_train_comb, y_train_comb)
        
        oof_xgb_probs[val_index] = m_xgb.predict_proba(X_val)[:, 1]
        oof_lgbm_probs[val_index] = m_lgbm.predict_proba(X_val)[:, 1]
    
    # Weight & Threshold 내부 탐색
    for w in weights:
        combined_probs = (w * oof_xgb_probs) + ((1 - w) * oof_lgbm_probs)
        for thr in thresholds:
            score = f1_score(y, (combined_probs >= thr).astype(int))
            all_results.append({'Ratio': ratio, 'Weight': w, 'Threshold': thr, 'F1': score})
            
            if score > best_overall['f1']:
                best_overall = {'f1': score, 'ratio': ratio, 'weight': w, 'thr': thr}

    print(f"📍 Ratio {ratio:.2f} 완료 | 현재 최고 F1: {best_overall['f1']:.4f} (W:{best_overall['weight']:.1f})")

# 4. 결과 정리 및 시각화 (Best Weight 기준 Heatmap)
df_res = pd.DataFrame(all_results)
best_w = best_overall['weight']

# 데이터를 다시 뽑고 소수점 정리
plot_df = df_res[df_res['Weight'] == best_w].copy()
plot_df['Ratio'] = plot_df['Ratio'].round(2)
plot_df['Threshold'] = plot_df['Threshold'].round(2)

# 피벗 테이블 생성
pivot_df = plot_df.pivot(index="Ratio", columns="Threshold", values="F1")

plt.figure(figsize=(16, 10))

# 틱 간격 조절: 모든 눈금을 표시하면 글자가 겹치므로 5개 간격으로 표시
# (간격은 데이터 양에 따라 조정하세요)
x_step = 5
y_step = 2

sns.heatmap(
    pivot_df, 
    annot=False, 
    cmap="YlGnBu", 
    cbar_kws={'label': 'F1-Score'},
    xticklabels=x_step, # X축 눈금 간격 설정
    yticklabels=y_step  # Y축 눈금 간격 설정
)

plt.title(f"Ensemble Heatmap (Best Weight: {best_w:.1f})\nGlobal Max F1: {best_overall['f1']:.4f} at R:{best_overall['ratio']:.2f}, T:{best_overall['thr']:.2f}", fontsize=15)
plt.xlabel("Classification Threshold", fontsize=12)
plt.ylabel("Synthetic Data Ratio (1:X)", fontsize=12)

# 축 레이블 회전 (가독성 향상)
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "v02_ensemble_clean_result.png"), dpi=300)
plt.show()

# 7. 최강 조합 요약 리포트
print("\n" + "="*60)
print("   🏆 HR ATTRITION PREDICTION: CHAMPION MODEL SETTING")
print("="*60)
print(f"  ✅ [Ratio]      : {best_overall['ratio']:.2f} (Synthetic Data Ratio)")
print(f"  ✅ [Threshold]  : {best_overall['thr']:.2f} (Classification Boundary)")
print(f"  ✅ [Weights]    : XGB {best_overall['weight']:.1f} : LGBM {1-best_overall['weight']:.1f}")
print("-" * 60)
print(f"  🔥 [Best F1]    : {best_overall['f1']:.4f}")
print("="*60)

# 최종 파라미터 변수화 (나중에 모델 저장 시 사용)
final_ratio = best_overall['ratio']
final_thr = best_overall['thr']
final_xgb_w = best_overall['weight']

# ============================================================
#    🏆 HR ATTRITION PREDICTION: CHAMPION MODEL SETTING
# ============================================================
#   ✅ [Ratio]      : 0.68 (Synthetic Data Ratio)
#   ✅ [Threshold]  : 0.51 (Classification Boundary)
#   ✅ [Weights]    : XGB 0.4 : LGBM 0.6
# ------------------------------------------------------------
#   🔥 [Best F1]    : 0.5581
# ============================================================