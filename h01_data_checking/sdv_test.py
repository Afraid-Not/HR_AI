import pandas as pd
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

# 1. 데이터 로드 (경로 에러 방지를 위해 r 붙임)
file_path = r"D:\hr_ai\_data\WA_Fn-UseC_-HR-Employee-Attrition.csv"
real_hr_df = pd.read_csv(file_path)

# 2. 메타데이터 추출
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=real_hr_df)

# 3. CTGAN 학습 및 생성 (enforce_privacy를 제거하고 기본 설정으로 진행)
print("--- CTGAN 학습 시작 ---")
ctgan = CTGANSynthesizer(metadata, epochs=500)
ctgan.fit(real_hr_df)
synthetic_ctgan = ctgan.sample(num_rows=1000)

# 4. TVAE 학습 및 생성
print("--- TVAE 학습 시작 ---")
tvae = TVAESynthesizer(metadata, epochs=500)
tvae.fit(real_hr_df)
synthetic_tvae = tvae.sample(num_rows=1000)

# 5. 성능 평가 비교
print("\n--- 성능 평가 리포트 ---")
report_ctgan = evaluate_quality(real_hr_df, synthetic_ctgan, metadata)
report_tvae = evaluate_quality(real_hr_df, synthetic_tvae, metadata)

print(f"\n[CTGAN 결과]")
print(f"Overall Score: {report_ctgan.get_score():.4f}")

print(f"\n[TVAE 결과]")
print(f"Overall Score: {report_tvae.get_score():.4f}")