from ast import Or
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("D:/hr_ai/_data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# print(df.columns)
# Index(['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
#        'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
#        'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
#        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
#        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
#        'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
#        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
#        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
#        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
#        'YearsWithCurrManager'],
#       dtype='object')

def columns_refiner(df, drop_cols, oe_cols, target_cols):
    df = df.drop(drop_cols, axis=1)
    oe = OrdinalEncoder()
    df[oe_cols] = oe.fit_transform(df[oe_cols])
    target = df[target_cols]
    df = df.drop(target_cols, axis=1)
    df = pd.concat([df, target], axis=1)
    
    return df

def apply_age_labeling(df):
    # 1. 나이 구간 나누기
    # 0~29: 0 (30 under)
    # 30~39: 1 (30s)
    # 40~49: 2 (40s)
    # 50 이상: 3 (50 over)
    
    def map_age(age):
        if age < 30: return 0
        elif age < 40: return 1
        elif age < 50: return 2
        else: return 3

    df['Age_Label'] = df['Age'].apply(map_age)
    df = df.drop('Age', axis=1)

    return df

def add_relative_salary_features(df):
    # 1. 직급(JobLevel)별 평균 월급 계산
    avg_salary_by_level = df.groupby('JobLevel')['MonthlyIncome'].transform('mean')
    
    # 2. 직급 평균 대비 본인의 급여 비율 (예: 0.8이면 평균보다 20% 적게 받음)
    df['Salary_Ratio_to_Level'] = df['MonthlyIncome'] / avg_salary_by_level
    
    # 3. (선택) 직급 대비 급여를 구간으로 나누기 (0: 저급여, 1: 평균, 2: 고급여)
    # 0.9 미만: 저보상 / 0.9~1.1: 적정 / 1.1 초과: 고보상
    df['Salary_Level_Tag'] = df['Salary_Ratio_to_Level'].apply(
        lambda x: 0 if x < 0.9 else (1 if x <= 1.1 else 2)
    )
    df = df.drop(['Salary_Ratio_to_Level'], axis=1)
    
    return df

def apply_distance_labeling(df):
    def map_distance(dist):
        if dist < 7: return 0
        elif dist < 15: return 1
        elif dist < 25: return 2
        else: return 3

    # 새로운 라벨 컬럼 생성
    df['Distance_Label'] = df['DistanceFromHome'].apply(map_distance)
    return df


drop_cols = [
    'DailyRate', 'EmployeeCount', 'EmployeeNumber',
    'HourlyRate', 'MonthlyRate', 'Over18', 'StandardHours',
    'YearsWithCurrManager'
]
oe_cols = [
    'BusinessTravel', 'EducationField', 'Gender',
    'JobRole', 'MaritalStatus', 'OverTime', 'Attrition',
    'Department'
]
target_cols = 'Attrition'
df['MonthlyIncome'] = df['MonthlyIncome']*1000
df = add_relative_salary_features(df)
df = apply_distance_labeling(df)

df = apply_age_labeling(df)
df = columns_refiner(df, drop_cols, oe_cols, target_cols)
print(df['Salary_Level_Tag'].head(10))

df.to_csv("./_data/preprocessed_data.csv", index=False)