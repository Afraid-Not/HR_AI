import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import joblib  # 모델 저장용
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils import resample
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings

# 1. 랜덤 시드 고정 (재현성 확보의 핵심)
def seed_everything(seed=120):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # XGB, LGBM은 모델 파라미터의 random_state로 제어
SEED=130
seed_everything(SEED)
warnings.filterwarnings('ignore')

# 2. 데이터 로드
real_set = pd.read_csv("./_data/preprocessed_data_v2.csv")
fake_set = pd.read_csv("./_data/augmented_dataset_10000_v2_score_90_96.csv")
X = real_set.drop('Attrition', axis=1)
y = real_set['Attrition']

save_dir = "./h05_v3_xgb_lgbm_ensemble/models"
if not os.path.exists(save_dir): os.makedirs(save_dir)

# 3. 탐색 범위 설정
ratios = np.arange(0.3, 0.81, 0.02)
weights = np.arange(0.1, 1.0, 0.1)
thresholds = np.arange(0.2, 0.81, 0.01)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
best_overall = {'f1': -1}
all_results = []

print(f"🚀 [3D Grid Search] 탐색 시작 seed:{SEED}...")

# 4. 메인 루프 (비율 탐색)
for ratio in ratios:
    oof_xgb_probs = np.zeros(len(y))
    oof_lgbm_probs = np.zeros(len(y))
    
    for train_index, val_index in skf.split(X, y):
        X_train_real, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train_real, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 증강 로직 (Seed 42 고정)
        train_0 = X_train_real[y_train_real == 0]
        train_1_real = X_train_real[y_train_real == 1]
        train_1_fake = fake_set[fake_set['Attrition'] == 1].drop('Attrition', axis=1)
        target_n = int(len(train_0) * ratio)
        needed_n = max(0, target_n - len(train_1_real))
        
        train_1_aug = resample(train_1_fake, n_samples=needed_n, replace=False, random_state=SEED)
        X_train_comb = pd.concat([train_0, train_1_real, train_1_aug])
        y_train_comb = [0]*len(train_0) + [1]*(len(train_1_real) + len(train_1_aug))
        
        # 모델 학습 (Random State 고정)
        m_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=SEED, eval_metric='logloss')
        m_lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=SEED, verbosity=-1)
        
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

    if round(ratio, 2) % 0.1 == 0:
        print(f"📍 Ratio {ratio:.2f} 완료 | 현재 최고 F1: {best_overall['f1']:.4f}")

# 5. [Champion] 최종 모델 재학습 및 저장
print("\n📦 최강 조합으로 최종 모델 패키징 중...")

# 베스트 설정으로 데이터 재구성
final_train_0 = real_set[real_set['Attrition'] == 0]
final_train_1_real = real_set[real_set['Attrition'] == 1]
final_train_1_fake = fake_set[fake_set['Attrition'] == 1].drop('Attrition', axis=1)

target_n = int(len(final_train_0) * best_overall['ratio'])
needed_n = max(0, target_n - len(final_train_1_real))
final_train_1_aug = resample(final_train_1_fake, n_samples=needed_n, replace=False, random_state=SEED)

X_final = pd.concat([final_train_0.drop('Attrition', axis=1), final_train_1_real.drop('Attrition', axis=1), final_train_1_aug])
y_final = [0]*len(final_train_0) + [1]*(len(final_train_1_real) + len(final_train_1_aug))

# 최종 모델 학습
final_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=SEED, eval_metric='logloss')
final_lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=SEED, verbosity=-1)
final_xgb.fit(X_final, y_final)
final_lgbm.fit(X_final, y_final)

# 모델 + 메타데이터 통합 저장
model_packet = {
    'xgb_model': final_xgb,
    'lgbm_model': final_lgbm,
    'ratio': best_overall['ratio'],
    'weight_xgb': best_overall['weight'],
    'threshold': best_overall['thr'],
    'f1_score': best_overall['f1'],
    'features': X.columns.tolist()
}
score_str = str(round(best_overall['f1'],4)).replace(".", "_")
joblib.dump(model_packet, os.path.join(save_dir, f"ensemble_v2_{SEED}_{score_str}.joblib"))

# 6. 최강 조합 리포트 출력
print("\n" + "="*60)
print(f"   🏆 HR ATTRITION PREDICTION: CHAMPION MODEL SAVED(SEED={SEED})")
print("="*60)
print(f"  ✅ [Ratio]      : {best_overall['ratio']:.2f}")
print(f"  ✅ [Threshold]  : {best_overall['thr']:.2f}")
print(f"  ✅ [Weights]    : XGB {best_overall['weight']:.1f} : LGBM {1-best_overall['weight']:.1f}")
print(f"  🔥 [Best F1]    : {best_overall['f1']:.4f}")
print(f"  📂 [Path]       : {save_dir}/ensemble_v2_{SEED}_{score_str}.joblib")
print("="*60)

# PS D:\hr_ai> & C:/Users/pc/anaconda3/envs/hr_ai/python.exe d:/hr_ai/h05_v3_xgb_lgbm_ensemble/v02_train.py
# 🚀 [3D Grid Search] 탐색 시작 (Seed Fixed: 42)...
# 📍 Ratio 0.40 완료 | 현재 최고 F1: 0.5427
# 📍 Ratio 0.80 완료 | 현재 최고 F1: 0.5581

# 📦 최강 조합으로 최종 모델 패키징 중...

# ============================================================
#    🏆 HR ATTRITION PREDICTION: CHAMPION MODEL SAVED
# ============================================================
#   ✅ [Ratio]      : 0.68
#   ✅ [Threshold]  : 0.51
#   ✅ [Weights]    : XGB 0.4 : LGBM 0.6
#   🔥 [Best F1]    : 0.5581
#   📂 [Path]       : ./h05_v3_xgb_lgbm_ensemble/champion_ensemble_model.joblib
# ============================================================

# 🚀 [3D Grid Search] 탐색 시작 seed:120...
# 📍 Ratio 0.40 완료 | 현재 최고 F1: 0.5440
# 📍 Ratio 0.80 완료 | 현재 최고 F1: 0.5673

# 📦 최강 조합으로 최종 모델 패키징 중...

# ============================================================
#    🏆 HR ATTRITION PREDICTION: CHAMPION MODEL SAVED(SEED=120)
# ============================================================
#   ✅ [Ratio]      : 0.70
#   ✅ [Threshold]  : 0.58
#   ✅ [Weights]    : XGB 0.1 : LGBM 0.9
#   🔥 [Best F1]    : 0.5673
#   📂 [Path]       : ./h05_v3_xgb_lgbm_ensemble/champion_ensemble_model_v2.joblib
# ============================================================

# 🚀 [3D Grid Search] 탐색 시작 seed:121...
# 📍 Ratio 0.40 완료 | 현재 최고 F1: 0.5731
# 📍 Ratio 0.80 완료 | 현재 최고 F1: 0.5835

# 📦 최강 조합으로 최종 모델 패키징 중...

# ============================================================
#    🏆 HR ATTRITION PREDICTION: CHAMPION MODEL SAVED(SEED=121)
# ============================================================
#   ✅ [Ratio]      : 0.52
#   ✅ [Threshold]  : 0.56
#   ✅ [Weights]    : XGB 0.2 : LGBM 0.8
#   🔥 [Best F1]    : 0.5835
#   📂 [Path]       : ./h05_v3_xgb_lgbm_ensemble/models/champion_ensemble_model_v2_121.joblib
# ============================================================

# 🚀 [3D Grid Search] 탐색 시작 seed:129...
# 📍 Ratio 0.40 완료 | 현재 최고 F1: 0.5853
# 📍 Ratio 0.80 완료 | 현재 최고 F1: 0.5992

# 📦 최강 조합으로 최종 모델 패키징 중...

# ============================================================
#    🏆 HR ATTRITION PREDICTION: CHAMPION MODEL SAVED(SEED=129)
# ============================================================
#   ✅ [Ratio]      : 0.42
#   ✅ [Threshold]  : 0.44
#   ✅ [Weights]    : XGB 0.1 : LGBM 0.9
#   🔥 [Best F1]    : 0.5992
#   📂 [Path]       : ./h05_v3_xgb_lgbm_ensemble/models/ensemble_v2_129_0_5992.joblib
# ============================================================
# PS D:\hr_ai> 