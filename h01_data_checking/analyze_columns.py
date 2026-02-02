import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv("D:\\hr_ai\\_data\\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# 결과를 저장할 리스트
results = []
results.append("=" * 80)
results.append("HR Employee Attrition 데이터셋 - 컬럼별 통계 및 선택지 분석")
results.append("=" * 80)
results.append(f"\n총 행 수: {len(df)}")
results.append(f"총 컬럼 수: {len(df.columns)}")
results.append("\n" + "=" * 80 + "\n")

# 각 컬럼별 분석
for col in df.columns:
    results.append(f"\n[{col}]")
    results.append("-" * 80)
    
    # 결측값 확인
    null_count = df[col].isnull().sum()
    results.append(f"결측값 개수: {null_count}")
    
    # 데이터 타입 확인
    dtype = df[col].dtype
    results.append(f"데이터 타입: {dtype}")
    
    # 고유값 개수
    unique_count = df[col].nunique()
    results.append(f"고유값 개수: {unique_count}")
    
    # 범주형 변수인 경우 (object 타입이거나 고유값이 적은 경우)
    if dtype == 'object' or unique_count <= 20:
        results.append(f"\n[선택지 목록]")
        
        # 각 선택지별 빈도수 (빈도순으로 정렬)
        value_counts = df[col].value_counts()
        for val, count in value_counts.items():
            percentage = (count / len(df)) * 100
            results.append(f"  - {val}: {count}개 ({percentage:.2f}%)")
    
    # 수치형 변수인 경우
    else:
        results.append(f"\n[통계 정보]")
        results.append(f"  평균: {df[col].mean():.2f}")
        results.append(f"  중앙값: {df[col].median():.2f}")
        results.append(f"  표준편차: {df[col].std():.2f}")
        results.append(f"  최소값: {df[col].min()}")
        results.append(f"  최대값: {df[col].max()}")
        results.append(f"  25% 분위수: {df[col].quantile(0.25):.2f}")
        results.append(f"  75% 분위수: {df[col].quantile(0.75):.2f}")
        
        # 고유값이 적은 수치형 변수는 선택지도 표시
        if unique_count <= 50:
            results.append(f"\n[고유값 목록]")
            unique_values = sorted(df[col].unique())
            results.append(f"  {', '.join(map(str, unique_values))}")

# 결과를 txt 파일로 저장
output_file = "D:\\hr_ai\\column_statistics.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

print(f"분석 완료! 결과가 '{output_file}'에 저장되었습니다.")
print(f"\n총 {len(df.columns)}개 컬럼 분석 완료")

