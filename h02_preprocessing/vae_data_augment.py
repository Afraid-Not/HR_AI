from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

import pandas as pd

df = pd.read_csv("./_data/preprocessed_data.csv")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=df)

tvae = TVAESynthesizer(
    metadata,
    embedding_dim=128,
    compress_dims=(128, 128),
    decompress_dims=(128, 128),
    batch_size=100,   # 데이터가 적으므로 배치를 줄여서 더 세밀하게 학습
    epochs=600,       # 학습 횟수를 충분히 확보
    loss_factor=3,    # 상관관계(Pair Trends) 복원력을 높임
    enable_gpu=True   # GPU가 있다면 속도가 빨라집니다
)
tvae.fit(df)
synthetic_tvae = tvae.sample(num_rows=10000)

report_tvae = evaluate_quality(df, synthetic_tvae, metadata)
score = report_tvae.get_score()

# 2. 100을 곱하고 소수점 2자리까지 반올림 (예: 86.85)
rounded_score = round(score * 100, 2)

print(f"Overall Score: {rounded_score}점")

# 3. 파일명 생성을 위해 문자열 변환 및 점(.)을 언더바(_)로 교체
# 예: 86.85 -> "86_85"
score_str = str(rounded_score).replace(".", "_")

# 4. 파일 저장
file_name = f"./_data/augmented_dataset_10000_score_{score_str}.csv"
synthetic_tvae.to_csv(file_name, index=False)

print(f"✅ 데이터 저장 완료: {file_name}")