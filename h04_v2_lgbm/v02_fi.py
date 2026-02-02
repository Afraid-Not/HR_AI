import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.utils import resample
import joblib
import os

# 1. 최적 파라미터 설정
BEST_RATIO = 0.68
BEST_THRESHOLD = 0.51
import warnings

# 경고 메시지 무시 (LGBM 특유의 정보성 로그 포함)
warnings.filterwarnings('ignore')

# 1. 데이터 로드
real_set = pd.read_csv("./_data/preprocessed_data.csv")
fake_set = pd.read_csv("./_data/augmented_dataset_10000_score_91_5.csv")
# 2. 최종 학습 데이터 구성 (Best Ratio 0.68 적용)
train_0 = real_set[real_set['Attrition'] == 0]
train_1_real = real_set[real_set['Attrition'] == 1]
train_1_fake = fake_set[fake_set['Attrition'] == 1].drop('Attrition', axis=1)

target_total_1 = int(len(train_0) * BEST_RATIO)
needed_n = max(0, target_total_1 - len(train_1_real))

# 가짜 데이터 보충
train_1_aug = resample(train_1_fake, n_samples=needed_n, replace=False, random_state=42)
X_final = pd.concat([train_0.drop('Attrition', axis=1), train_1_real.drop('Attrition', axis=1), train_1_aug])
y_final = [0]*len(train_0) + [1]*(len(train_1_real) + len(train_1_aug))

# 3. 최종 모델 학습
final_lgbm = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,
    random_state=42,
    importance_type='gain', # 정보 이득(Gain) 기준
    verbosity=-1
)
final_lgbm.fit(X_final, y_final)

# 4. Feature Importance 시각화
ft_importances = pd.Series(final_lgbm.feature_importances_, index=X_final.columns)
top_15_ft = ft_importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(12, 8))
sns.barplot(x=top_15_ft.values, y=top_15_ft.index, palette='magma')
plt.title(f'LGBM Feature Importance (Gain) - Best Ratio {BEST_RATIO}', fontsize=15)
plt.xlabel('Total Gain')
plt.show()

# 5. 모델 및 메타데이터 저장
model_path = "./h04_v2_lgbm/final_attrition_model.joblib"
model_data = {
    'model': final_lgbm,
    'threshold': BEST_THRESHOLD,
    'features': X_final.columns.tolist()
}
joblib.dump(model_data, model_path)
print(f"✅ 최종 모델이 '{model_path}'에 저장되었습니다. (Threshold: {BEST_THRESHOLD})")