import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text

# 1. 리뷰 데이터 불러오기
df = pd.read_csv("airbnbview/cluster_1_reviews.csv")
texts = df['content'].dropna().astype(str).tolist()

# 2. 벡터라이저 (2-gram + 불용어 제거)
custom_stopwords = text.ENGLISH_STOP_WORDS.union([
    'app', 'airbnb', 'use', 'my', 'you', 'they', 'we', 'can', 'not', 'are', 'but', 'me', 'have'
])
vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words=list(custom_stopwords))
X = vectorizer.fit_transform(texts)
terms = vectorizer.get_feature_names_out()

# 3. 공출현 행렬 및 그래프 생성
co_occur = (X.T @ X)
co_occur.setdiag(0)
G = nx.from_scipy_sparse_array(co_occur)

# 4. 중심성 계산
centrality = nx.degree_centrality(G)

# 5. 중심성 높은 노드 30개 추출
top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:30]
subgraph_nodes = [n[0] for n in top_nodes]
top_terms = [terms[n] for n in subgraph_nodes]

# 6. 서브그래프 생성
G_sub = G.subgraph(subgraph_nodes)
labels = {n: terms[n] for n in G_sub.nodes()}
node_size = [centrality[n]*3000 for n in G_sub.nodes()]

# 7. 네트워크 시각화
pos = nx.spring_layout(G_sub, k=0.3)
plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G_sub, pos, node_size=node_size, node_color='skyblue')
nx.draw_networkx_edges(G_sub, pos, alpha=0.5)
nx.draw_networkx_labels(G_sub, pos, labels, font_size=10)
plt.title("Keyword Network (Top Central Terms in Cluster 1)", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()

print("✅ 네트워크 그래프가 성공적으로 시각화되었습니다.")
