import pandas as pd
from textblob import TextBlob

# 리뷰 데이터 불러오기
df = pd.read_csv("airbnbview/airbnb_reviews_238_per_quarter.csv")

# 감성 점수 계산 함수
def get_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# 감성 점수 컬럼 추가
df['sentiment'] = df['content'].apply(get_sentiment)

# 결과 확인
print(df[['content', 'sentiment']].head())

# 저장도 가능
df.to_csv("airbnbview/airbnb_reviews_sentiment_238.csv", index=False)
print("✅ 감성 분석이 완료되어 airbnb_reviews_sentiment.csv에 저장되었습니다.")
# 감성 점수 시각화
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
# Mac에서 기본 한글 폰트 지정
plt.rcParams['font.family'] = 'AppleGothic'
# 음수 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False
# 감성 점수 분포 시각화
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment'], bins=30, kde=True)
plt.title('Airbnb 리뷰 감성 점수 분포')
plt.xlabel('감성 점수')
plt.ylabel('빈도수')
plt.xlim(-1, 1)
plt.axvline(x=0, color='red', linestyle='--', label='중립')
plt.legend()
plt.show()