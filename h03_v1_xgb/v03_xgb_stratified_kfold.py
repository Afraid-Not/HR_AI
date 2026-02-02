from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.utils import resample

# 1. 데이터 로드
real_set = pd.read_csv("./_data/preprocessed_data.csv")
fake_set = pd.read_csv("./_data/augmented_dataset_10000_score_91_5.csv")
# 1. 실제 데이터 준비 (real_set만 사용)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import resample

# 1. 실제 데이터 준비
X = real_set.drop('Attrition', axis=1)
y = real_set['Attrition']

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_reports = []
avg_f1_scores = []

print("🚀 Stratified 5-Fold Cross-Validation 시작\n")

for i, (train_index, val_index) in enumerate(skf.split(X, y)):
    # 폴드 분리
    X_train_real, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_real, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # 학습셋에만 가짜 퇴사자 데이터 보충 (1:1 비율)
    train_0 = X_train_real[y_train_real == 0]
    train_1_real = X_train_real[y_train_real == 1]
    train_1_fake = fake_set[fake_set['Attrition'] == 1].drop('Attrition', axis=1)
    
    needed_n = len(train_0) - len(train_1_real)
    train_1_augmented = resample(train_1_fake, n_samples=needed_n, replace=False, random_state=42)
    
    X_train_combined = pd.concat([train_0, train_1_real, train_1_augmented])
    y_train_combined = [0]*len(train_0) + [1]*(len(train_1_real) + len(train_1_augmented))
    
    # 모델 학습
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, eval_metric='logloss')
    model.fit(X_train_combined, y_train_combined)
    
    # 평가
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    avg_f1_scores.append(f1)
    
    print(f"--- Fold {i+1} 결과 ---")
    print(report)
    print("-" * 30)

print(f"\n✅ 5-Fold 평균 Attrition F1-Score: {np.mean(avg_f1_scores):.4f}")


#  Attrition F1-Score: 0.5187