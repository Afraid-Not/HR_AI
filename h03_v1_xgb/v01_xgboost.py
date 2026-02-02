from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import pandas as pd
import numpy as np

real_set = pd.read_csv("./_data/preprocessed_data.csv")
fake_set = pd.read_csv("./_data/augmented_dataset_10000_score_91_5.csv")

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score

# 1. 실제 데이터에서 평가용(Test) 20% 미리 떼어놓기
real_train, real_test = train_test_split(real_set, test_size=0.2, random_state=42, stratify=real_set['Attrition'])

# 2. 학습용 데이터 구성 (방법 B: 실제 80% + TVAE 데이터)
# synthetic_tvae는 아까 만드신 10,000건 데이터
combined_train = pd.concat([real_train, fake_set], axis=0)

# 3. 모델 학습 (XGBoost)
model = XGBClassifier()
model.fit(combined_train.drop('Attrition', axis=1), combined_train['Attrition'])

# 4. '순수 실제 데이터(real_test)'로 평가
predictions = model.predict(real_test.drop('Attrition', axis=1))

print("--- 실제 데이터 기반 최종 평가 결과 ---")
print(classification_report(real_test['Attrition'], predictions))

#               precision    recall  f1-score   support

#          0.0       0.90      0.94      0.92       247
#          1.0       0.57      0.45      0.50        47

#     accuracy                           0.86       294
#    macro avg       0.73      0.69      0.71       294
# weighted avg       0.85      0.86      0.85       294