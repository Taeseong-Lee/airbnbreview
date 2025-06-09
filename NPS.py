import pandas as pd

# 1) 리뷰 데이터 로드 & 약식 NPS 카테고리 계산
df = pd.read_csv("airbnbview/cluster_1_reviews.csv")  # 부정/중립 클러스터 예시
def star_to_nps(score):
    if score >= 4:
        return "Promoter"
    elif score == 3:
        return "Passive"
    else:
        return "Detractor"
df['nps_cat'] = df['score'].apply(star_to_nps)

# 2) 기본 NPS, 집단별 비율 계산
counts = df['nps_cat'].value_counts()
total = counts.sum()
promoters = counts.get('Promoter', 0)
detractors = counts.get('Detractor', 0)
passives = counts.get('Passive', 0)

base_nps = (promoters - detractors) / total * 100
print(f"Baseline Approximate NPS: {base_nps:.1f}")

# 3) 솔루션별 부정집단 유사도 (예시)
#    sim_neg 는 react.py로 구한 “부정 클러스터와 솔루션 키워드 유사도”
solutions = {
    "Host Trust Score":         {"sim_neg": 0.262},
    "Localized Risk Index":     {"sim_neg": 0.058},
    "Trust Insights UI":        {"sim_neg": 0.282},
    "AI Copilot assistant":     {"sim_neg": 0.133},
    "Sponsored Listings":       {"sim_neg": 0.238},
    "Trust Insights Pro":       {"sim_neg": 0.260},
    "Data Pipeline Moderniz’n": {"sim_neg": 0.054},
    "Rapid A/B Guardrails":     {"sim_neg": 0.077},
    "Inference-on-Edge":        {"sim_neg": 0.086},
    "Data-Share Partnerships":  {"sim_neg": 0.162},
}

# 4) 시뮬레이션 함수: sim_neg 비율만큼 Detractor→Promoter 전환
def simulate_nps(sim_neg):
    # 전환 대상 Detractors
    converted = detractors * sim_neg
    new_prom = promoters + converted
    new_det = detractors - converted
    # Passive는 그대로
    return (new_prom - new_det) / total * 100

# 5) 모든 솔루션에 대해 예측 NPS 계산
results = []
for name, v in solutions.items():
    pred_nps = simulate_nps(v["sim_neg"])
    delta = pred_nps - base_nps
    results.append((name, pred_nps, delta))

# 6) 결과 출력
res_df = pd.DataFrame(results, columns=["Solution","Predicted NPS","Δ vs Base"])
res_df = res_df.sort_values("Predicted NPS", ascending=False).round(1)
print("\nPredicted NPS after solution deployment:\n", res_df)
