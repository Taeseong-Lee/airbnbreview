import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

# 1. 클러스터 파일 불러오기
df0 = pd.read_csv("airbnbview/cluster_0_reviews.csv")
df1 = pd.read_csv("airbnbview/cluster_1_reviews.csv")

# 불용어 확장
# 수정된 코드
custom_stopwords = list(text.ENGLISH_STOP_WORDS.union([
    'app', 'airbnb', 'use', 'my', 'you', 'they', 'we', 'can', 'not', 'are', 'but', 'me', 'have'
]))

vectorizer = CountVectorizer(stop_words=custom_stopwords)


# 3. 벡터화 및 단어 빈도 계산
def get_top_keywords(texts, top_n=15):
    X = vectorizer.fit_transform(texts)
    sum_words = X.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return words_freq[:top_n]

# 4. 결과 도출
top_words_0 = get_top_keywords(df0['content'].astype(str))
top_words_1 = get_top_keywords(df1['content'].astype(str))


# 5. 시각화 함수
def plot_keywords(word_freq, cluster_id):
    words, counts = zip(*word_freq)
    # Mac에서 기본 한글 폰트 지정
    plt.rcParams['font.family'] = 'AppleGothic'
    # 음수 부호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,5))
    sns.barplot(x=list(counts), y=list(words), palette="coolwarm")
    plt.title(f"Top Keywords in Cluster {cluster_id}")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.show()

# 6. 시각화 출력
plot_keywords(top_words_0, 0)
plot_keywords(top_words_1, 1)
