import pandas as pd
import matplotlib.pyplot as plt

# 1. 리뷰 데이터 불러오기 (sentiment 컬럼이 있는 CSV)
df = pd.read_csv("airbnbview/airbnb_reviews_sentiment_238.csv", parse_dates=["at"])
df["quarter"] = df["at"].dt.to_period("Q").dt.to_timestamp()

# 2. 분기별 부정 sentiment 비율 계산 (sentiment < 0)
neg_share = (
    df.groupby("quarter")["sentiment"]
      .apply(lambda x: (x < 0).mean() * 100)
      .rename("neg_sentiment_pct")
)

# 3. 분기별 OPEX 데이터 정의 (단위: M USD)
opex_data = {
    "2020-03-31": 1167, "2020-06-30":  918, "2020-09-30":  924, "2020-12-31": 3959,
    "2021-03-31": 1334, "2021-06-30": 1386, "2021-09-30": 1385, "2021-12-31": 1457,
    "2022-03-31": 1514, "2022-06-30": 1735, "2022-09-30": 1681, "2022-12-31": 1667,
    "2023-03-31": 1823, "2023-06-30": 1961, "2023-09-30": 1901, "2023-12-31": 2714,
    "2024-03-31": 2041, "2024-06-30": 2251, "2024-09-30": 2207, "2024-12-31": 2050,
    "2025-03-31": 2234
}
opex_df = pd.DataFrame({
    "quarter": pd.to_datetime(list(opex