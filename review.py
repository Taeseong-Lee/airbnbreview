# 기존 코드
from google_play_scraper import reviews
import pandas as pd

result, _ = reviews(
    'com.airbnb.android',  # Airbnb 앱의 패키지명
    lang='en',
    country='us',
    count=1000
)

df = pd.DataFrame(result)
df = df[['userName', 'score', 'content', 'at']]

# ✅ CSV로 저장하는 코드 추가
df.to_csv("airbnbview/airbnb_reviews.csv", index=False)
print("리뷰 데이터가 airbnb_reviews.csv로 저장되었습니다.")