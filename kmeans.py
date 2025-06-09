from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
df = pd.read_csv("airbnbview/airbnb_reviews_sentiment_238.csv")

# 감성 점수만 추출
X = df[['sentiment']]

# KMeans 클러스터링 (군집 수는 2로 시작)
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# 클러스터별로 CSV 저장
for cluster_num in df['cluster'].unique():
    cluster_df = df[df['cluster'] == cluster_num]
    filename = f"airbnbview/cluster_{cluster_num}_reviews_Q.csv"
    cluster_df.to_csv(filename, index=False)
    print(f"✅ {filename} 저장 완료")
    
# Mac에서 기본 한글 폰트 지정
plt.rcParams['font.family'] = 'AppleGothic'
# 음수 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 시각화
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='sentiment', hue='cluster', bins=30, kde=True, palette='Set2')
plt.title("감성 점수 기반 클러스터링 결과 (KMeans)")
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.show()
# 클러스터 중심 시각화
centers = kmeans.cluster_centers_
plt.axvline(x=centers[0][0], color='red', linestyle='--', label='Cluster 1 Center')
plt.axvline(x=centers[1][0], color='blue', linestyle='--', label='Cluster 2 Center')
plt.legend()
plt.title("Cluster Centers")
plt.xlabel("Sentiment Score")
plt.ylabel("Count")
plt.show()

