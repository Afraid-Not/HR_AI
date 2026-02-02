import pandas as pd

df = pd.read_csv("D:\hr_ai\_data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
# print(df.columns)
# Index(['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
# 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
# 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
# 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
# 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
# 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating',
# 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
# 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
# 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
# 'YearsWithCurrManager'],
print(df.head(5))

