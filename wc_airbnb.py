import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. 리뷰 CSV 불러오기
df0 = pd.read_csv("airbnbview/cluster_0_reviews.csv")
df1 = pd.read_csv("airbnbview/cluster_1_reviews.csv")

# 2. 모든 리뷰를 하나의 문자열로 합치기
text_0 = " ".join(df0['content'].dropna().astype(str))
text_1 = " ".join(df1['content'].dropna().astype(str))

# 3. 불용어 설정 (기본 + 사용자 지정)
from sklearn.feature_extraction import text
stopwords = set(text.ENGLISH_STOP_WORDS)
stopwords.update(['app', 'airbnb', 'use', 'my', 'you', 'they', 'we', 'can', 'not', 'are', 'but', 'me', 'have'])

# 4. 워드클라우드 생성 함수
def make_wordcloud(text, title):
    wc = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        max_words=100,
        colormap='viridis'
    ).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

# 5. 각 클러스터 시각화
make_wordcloud(text_0, "WordCloud for Cluster 0 (Positive Users)")
make_wordcloud(text_1, "WordCloud for Cluster 1 (Neutral/Negative Users)")
