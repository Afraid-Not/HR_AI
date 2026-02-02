import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample

# 1. 데이터 로드
real_set = pd.read_csv("./_data/preprocessed_data.csv")
fake_set = pd.read_csv("./_data/augmented_dataset_10000_score_91_5.csv")

# 2. 평가용 데이터(Test) 20% 분리 (stratify 필수)
real_train, real_test = train_test_split(
    real_set, test_size=0.2, random_state=42, stratify=real_set['Attrition']
)

# 3. [핵심] 1:1 비율 맞추기 전략
# 실제 재직자(0) 수만큼 퇴사자(1) 데이터를 확보합니다.
train_0 = real_train[real_train['Attrition'] == 0]
train_1_real = real_train[real_train['Attrition'] == 1]
train_1_fake = fake_set[fake_set['Attrition'] == 1] # 가짜 데이터에서는 '퇴사자'만 추출

# 부족한 퇴사자 샘플 수 계산
target_n = len(train_0)
current_n = len(train_1_real)
needed_n = target_n - current_n

# 가짜 데이터에서 부족한 만큼 샘플링
train_1_augmented = resample(
    train_1_fake, 
    replace=False,    # 10,000건 중 일부만 쓰므로 중복 없이
    n_samples=needed_n, 
    random_state=42
)

# 최종 학습셋: 실제 재직자 + 실제 퇴사자 + 가짜 퇴사자 (정확히 1:1)
combined_train = pd.concat([train_0, train_1_real, train_1_augmented], axis=0)

print(f"✅ 학습 데이터 구성 완료")
print(f"- 재직자(0): {len(train_0)}명")
print(f"- 퇴사자(1): {len(train_1_real) + len(train_1_augmented)}명 (실제:{len(train_1_real)} + 가짜:{len(train_1_augmented)})")

# 4. 모델 학습 (파라미터 살짝 튜닝)
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(combined_train.drop('Attrition', axis=1), combined_train['Attrition'])

# 5. 순수 실제 데이터로 평가
X_test = real_test.drop('Attrition', axis=1)
y_test = real_test['Attrition']
predictions = model.predict(X_test)

print("\n--- [1:1 오버샘플링 적용] 최종 평가 결과 ---")
print(classification_report(y_test, predictions))

import matplotlib.pyplot as plt
import seaborn as sns

# 변수 중요도 데이터프레임 생성
feature_names = combined_train.drop('Attrition', axis=1).columns
importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 시각화 (상위 15개)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='viridis')
plt.title('Top 15 Feature Importance for Attrition Prediction', fontsize=15)
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# 상세 수치 출력
print(feature_importance_df.head(15))


#       --- [1:1 오버샘플링 적용] 최종 평가 결과 ---
#               precision    recall  f1-score   support

#          0.0       0.93      0.86      0.89       247
#          1.0       0.48      0.68      0.56        47

#     accuracy                           0.83       294
#    macro avg       0.71      0.77      0.73       294
# weighted avg       0.86      0.83      0.84       294