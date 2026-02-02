# HR Employee Attrition Prediction

직원 이직 예측을 위한 머신러닝 프로젝트입니다. XGBoost와 LightGBM을 앙상블하여 최종 모델을 구축했습니다.

## 📋 프로젝트 개요

이 프로젝트는 HR 데이터를 활용하여 직원의 이직(Attrition) 여부를 예측하는 것을 목표로 합니다. 
데이터 불균형 문제를 해결하기 위해 데이터 증강 기법을 적용하고, 여러 모델을 실험하여 최적의 성능을 달성했습니다.

## 🎯 주요 성과

- **최종 모델 성능**: F1 Score **0.5992** (Seed 129)
- **모델 구조**: XGBoost + LightGBM Ensemble
- **데이터 증강**: VAE 기반 합성 데이터 생성으로 클래스 불균형 해결

## 📁 프로젝트 구조

```
hr_ai/
├── _data/                          # 데이터 파일
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv  # 원본 데이터
│   ├── preprocessed_data_v2.csv               # 전처리된 데이터
│   └── augmented_dataset_*.csv                # 증강된 데이터
│
├── h01_data_checking/              # 데이터 탐색 및 분석
│   ├── check_data.py               # 데이터 기본 확인
│   ├── analyze_columns.py          # 컬럼별 통계 분석
│   └── column_statistics.txt       # 컬럼 통계 결과
│
├── h02_preprocessing/              # 데이터 전처리 및 증강
│   ├── preprocessing_v2.py        # 데이터 전처리 (v2)
│   └── vae_data_augment_v2.py     # VAE 기반 데이터 증강
│
├── h03_v1_xgb/                     # XGBoost 모델 실험
│   ├── v01_xgboost.py              # 기본 XGBoost
│   ├── v02_xgb_resample.py        # 리샘플링 적용
│   ├── v03_xgb_stratified_kfold.py # Stratified K-Fold
│   └── v06_grid_search.py          # 하이퍼파라미터 튜닝
│
├── h04_v2_lgbm/                    # LightGBM 모델 실험
│   ├── v01_lgbm.py                 # 기본 LightGBM
│   └── final_attrition_model.joblib # 최종 모델
│
└── h05_v3_xgb_lgbm_ensemble/       # 앙상블 모델 (최종)
    ├── v02_train.py                # 3D Grid Search 앙상블 학습
    └── models/                     # 저장된 모델들
        └── ensemble_v2_129_0_5992.joblib  # 최고 성능 모델
```

## 🚀 사용 방법

### 1. 환경 설정

```bash
# conda 가상환경 생성 및 활성화
conda create -n hr_ai python=3.8
conda activate hr_ai

# 필요한 패키지 설치
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn joblib
```

### 2. 데이터 전처리

```bash
python h02_preprocessing/preprocessing_v2.py
```

전처리 과정:
- 불필요한 컬럼 제거 (DailyRate, EmployeeCount, EmployeeNumber 등)
- 범주형 변수 인코딩 (OrdinalEncoder)
- 연속형 변수 스케일링 (StandardScaler)
- 특징 엔지니어링:
  - 나이 구간화 (Age_Label)
  - 거리 구간화 (Distance_Label)
  - 직급 대비 급여 비율 (Salary_Level_Tag)

### 3. 데이터 증강

```bash
python h02_preprocessing/vae_data_augment_v2.py
```

VAE(Variational Autoencoder)를 사용하여 소수 클래스(이직자) 데이터를 증강합니다.

### 4. 모델 학습

```bash
python h05_v3_xgb_lgbm_ensemble/v02_train.py
```

**앙상블 모델 학습 과정**:
1. **3D Grid Search**: Ratio, Weight, Threshold를 탐색
   - Ratio: 0.3 ~ 0.8 (증강 데이터 비율)
   - Weight: 0.1 ~ 0.9 (XGBoost 가중치)
   - Threshold: 0.2 ~ 0.8 (예측 임계값)
2. **Stratified K-Fold**: 5-fold 교차 검증으로 성능 평가
3. **최적 조합 선택**: F1 Score가 가장 높은 조합 선택
4. **최종 모델 저장**: 모델과 메타데이터를 joblib 형식으로 저장

## 📊 모델 성능

| Seed | Ratio | Threshold | Weight (XGB:LGBM) | F1 Score |
|------|-------|-----------|------------------|----------|
| 121  | 0.52  | 0.56      | 0.2 : 0.8        | 0.5835   |
| 129  | 0.42  | 0.44      | 0.1 : 0.9        | **0.5992** |

## 🔧 주요 특징

### 데이터 전처리
- **나이 구간화**: 0~29, 30~39, 40~49, 50+ 세 구간으로 분류
- **거리 구간화**: 출퇴근 거리를 4개 구간으로 분류
- **급여 상대화**: 직급별 평균 급여 대비 비율 계산

### 모델링
- **데이터 증강**: VAE 기반 합성 데이터로 클래스 불균형 해결
- **앙상블**: XGBoost와 LightGBM의 예측 확률을 가중 평균
- **임계값 최적화**: F1 Score를 최대화하는 임계값 탐색

## 📝 데이터셋 정보

- **원본 데이터**: `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **총 행 수**: 1,470개
- **총 컬럼 수**: 35개
- **타겟 변수**: Attrition (Yes/No)
- **클래스 불균형**: 이직자 16.12% vs 비이직자 83.88%

주요 특징:
- Age, DistanceFromHome, MonthlyIncome 등 수치형 변수
- Department, JobRole, EducationField 등 범주형 변수
- JobSatisfaction, EnvironmentSatisfaction 등 만족도 점수

## 📦 모델 저장 형식

저장된 모델은 다음과 같은 구조를 가집니다:

```python
{
    'xgb_model': XGBClassifier,
    'lgbm_model': LGBMClassifier,
    'ratio': float,           # 증강 데이터 비율
    'weight_xgb': float,      # XGBoost 가중치
    'threshold': float,       # 예측 임계값
    'f1_score': float,        # F1 Score
    'features': list          # 특징 이름 리스트
}
```

## 🔍 실험 버전 히스토리

- **v1 (XGBoost)**: 기본 XGBoost 모델 실험
- **v2 (LightGBM)**: LightGBM 모델로 전환
- **v3 (Ensemble)**: XGBoost + LightGBM 앙상블로 최종 성능 달성

## 📄 라이선스

이 프로젝트는 교육 및 연구 목적으로 작성되었습니다.

## 👤 작성자

재현
