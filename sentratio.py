import pandas as pd
import matplotlib.pyplot as plt

# 1. 리뷰 데이터 로드 (sentiment 칼럼: -1~+1 사이 값)
df = pd.read_csv("airbnbview/airbnb_reviews_sentiment_238.csv", parse_dates=["at"])

# 2. 분기 컬럼 생성
df["quarter"] = df["at"].dt.to_period("Q").dt.to_timestamp()

# 3. 분기별 전체 리뷰 수와 부정 리뷰 수 집계
grp = df.groupby("quarter")["sentiment"]
summary = pd.DataFrame({
    "total_reviews": grp.count(),
    "neg_reviews":  grp.apply(lambda x: (x < 0).sum())
})

# 4. 부정 비율 계산
summary["neg_pct"] = summary["neg_reviews"] / summary["total_reviews"] * 100

# 5. 모든 분기에 대해 0으로 채우기 (리뷰가 없는 분기도 0%)
all_quarters = pd.period_range(summary.index.min(), summary.index.max(), freq="Q").to_timestamp()
summary = summary.reindex(all_quarters, fill_value=0)

# 6. 시각화
plt.figure(figsize=(10, 5))
plt.plot(summary.index, summary["neg_pct"], marker="o", linestyle="-", color="orange")
plt.title("Quarterly Negative Sentiment Share")
plt.xlabel("Quarter")
plt.ylabel("Negative Sentiment (%)")
plt.xticks(rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.sho