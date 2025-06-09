import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# 1) 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2) 분기별 리뷰 읽어오기
#    csv에는 ['userName','score','content','at','sentiment','quarter'] 컬럼이 있다고 가정
df = pd.read_csv("airbnbview/airbnb_reviews_sentiment_238.csv", parse_dates=['at'])
# quarter 컬럼 포맷: '2020Q1', '2020Q2', ... '2025Q1'

# 3) 신기능 키워드 임베딩(예: Host Trust Score, Localized Risk Index 등)
new_features = [
    "Host Trust Score", "Localized Risk Index",
    "Trust Insights UI", "AI Copilot assistant",
    "Inference-On-Edge", "Data-Share Partnerships"

    # ... 나머지
]
feat_emb = model.encode(new_features, convert_to_numpy=True, normalize_embeddings=True)

# 4) 리뷰 텍스트 임베딩 (batch로 하면 속도 빠름)
texts = df['content'].astype(str).tolist()
embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# 5) 각 리뷰별로 가장 가까운 신기능 CosineSimilarity 계산
#    (feat_emb @ review_emb) 최대값을 그 리뷰의 'feature_sim'으로
sims = embs @ feat_emb.T     # (n_reviews, n_features)
df['feature_sim'] = sims.max(axis=1)

# 6) 분기별 평균 CosineSimilarity (신기능에 대한 반응 강도 지표)
q_sim = df.groupby('quarter')['feature_sim'].mean().rename('avg_feat_sim')

# 7) 분기별 OPEX 읽어오기
opex = pd.read_csv("airbnbview/opex.csv", parse_dates=['date'])
# date 컬럼: 실제 날짜. quarter로 변환
opex['quarter'] = opex['date'].dt.to_period('Q').astype(str)
opex_q = opex.groupby('quarter')['opex_musd'].sum()

# 8) 두 시계열 합치기
data = pd.concat([opex_q, q_sim], axis=1).dropna()
X = data[['avg_feat_sim']].values
y = data['opex_musd'].values

# 9) 회귀분석 (statsmodels로 p-value 등 출력)
X_sm = sm.add_constant(X)
model_ols = sm.OLS(y, X_sm).fit()
print(model_ols.summary())

# 10) 회귀계수로 신기능 도입 시 예측
#     예: feature_sim이 Δ0.05 증가할 때 opex가 Δcoef*0.05 증가
coef = model_ols.params[1]
print(f"신기능 반응 강도가 0.05 오르면 O