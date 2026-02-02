from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

import pandas as pd

df = pd.read_csv("D:\hr_ai\_data\WA_Fn-UseC_-HR-Employee-Attrition.csv")
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
synthetic_tvae.to_csv("D:/hr_ai/_data/augmented_dataset_10000.csv")

report_tvae = evaluate_quality(df, synthetic_tvae, metadata)

print(f"Overall Score: {report_tvae.get_score():.4f}")