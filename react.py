# react.py
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# 1) 모델 로드
model = SentenceTransformer('all-mpnet-base-v2')

# 2) 클러스터 별 리뷰 로딩
pos_df = pd.read_csv("airbnbview/cluster_0_reviews.csv")
neg_df = pd.read_csv("airbnbview/cluster_1_reviews.csv")

# 3) 센트로이드 계산 함수
def compute_centroid(df):
    texts = df['content'].dropna().astype(str).tolist()
    embeds = model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
    return torch.mean(embeds, dim=0)

cent_pos = compute_centroid(pos_df)
cent_neg = compute_centroid(neg_df)

# 4) 테스트할 새 기능 리스트
new_features = [
    "Host Trust Score",
    "Localized Risk Index",
    "Trust Insights UI",
    "AI Copilot assistant"
    "Sponsored Listings with Trust Floor",
    "Trust Insights Pro : Saas for Hosts",
    "Data Pipeline Modernization",
    "Rapid-A/B Guardrails",
    "Inference-on-Edge : Cloud to Edge",
    "Data-Share Partnerships with governments"
]
feat_embeds = model.encode(new_features, convert_to_tensor=True, normalize_embeddings=True)

# 5) 코사인 유사도로 각 클러스터 반응 예측
sim_pos = torch.nn.functional.cosine_similarity(feat_embeds, cent_pos.unsqueeze(0))
sim_neg = torch.nn.functional.cosine_similarity(feat_embeds, cent_neg.unsqueeze(0))

# 6) 결과 출력
for feat, p, n in zip(new_features, sim_pos, sim_neg):
    print(f"{feat:25s}  긍정 유사도: {p:.3f}   부정 유사도: {n:.3f}")
