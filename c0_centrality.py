import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer

# 클러스터 1 불러오기
df = pd.read_csv("airbnbview/cluster_0_reviews.csv")
texts = df['content'].dropna().astype(str).tolist()

# 1. 2-gram 벡터라이저
vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english')
X = vectorizer.fit_transform(texts)

# 2. 단어쌍 리스트 추출
terms = vectorizer.get_feature_names_out()
co_occur_matrix = X.T @ X
co_occur_matrix.setdiag(0)

# 3. 네트워크 그래프 생성
G = nx.from_scipy_sparse_array(co_occur_matrix)

# 4. 중심성 계산
centrality = nx.degree_centrality(G)
top_keywords = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:15]

# 5. 결과 출력
print("🔑 Top 15 Central Terms in Cluster 0:")
for term, score in top_keywords:
    print(f"{terms[term]}: {score:.4f}")
