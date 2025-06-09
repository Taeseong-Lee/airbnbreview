from google_play_scraper import reviews
import pandas as pd
from datetime import datetime

APP_ID       = "com.airbnb.android"
PER_PAGE     = 200         # 한 번에 가져오는 리뷰 수
TARGET_PER_Q = 238         # 분기당 목표 리뷰 수
START_DATE   = datetime(2020,1,1)
END_DATE     = datetime(2025,4,1)  # 2025Q1 포함

# 1) 가능한 많은 리뷰 스크래핑 (일단 넉넉히 2배수 정도 가져옴)
all_reviews = []
token       = None
while True:
    batch, token = reviews(
        APP_ID,
        lang               = "en",
        country            = "us",
        count              = PER_PAGE,
        continuation_token = token
    )
    all_reviews.extend(batch)

    # 가장 오래된 리뷰가 START_DATE 이전이면 멈춤
    dates = [r["at"] for r in batch]
    if dates and min(dates) < START_DATE:
        break
    if not token:
        break

# 2) DataFrame 변환 & 기간 필터
df = pd.DataFrame(all_reviews)
df["at"] = pd.to_datetime(df["at"])
df = df[(df["at"] >= START_DATE) & (df["at"] < END_DATE)]

# 3) 분기 컬럼 생성
df["quarter"] = df["at"].dt.to_period("Q")

# 4) 분기 리스트 생성 (2020Q1~2025Q1) 후, 분기별 샘플링
quarters = pd.period_range("2020Q1","2025Q1",freq="Q")

samples = []
for q in quarters:
    group = df[df["quarter"] == q]
    # 해당 분기에 리뷰가 충분하면 238건, 모자라면 가능한 만큼만
    n = min(TARGET_PER_Q, len(group))
    if n > 0:
        samples.append(group.sample(n=n, random_state=42))

# 5) 최종 데이터프레임 & 저장
final_df = pd.concat(samples).reset_index(drop=True)
print(f"최종 리뷰 개수: {len(final_df)}")  # 최대 238*21 = 4998건

final_df.to_csv("airbnbview/airbnb_reviews_238_per_quarter.csv", index=False)
print("✅ 분기별 238건씩 추출된 리뷰가 저장되었습니다.")
