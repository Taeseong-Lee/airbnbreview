import pandas as pd

# 1) 리뷰 데이터 불러오기
df_neg = pd.read_csv("airbnbview/cluster_1_reviews.csv")  # 부정 클러스터 리뷰

# 2) 'refund', 'cancel', 'refus' 등의 키워드 포함 비율 계산 (baseline)
refund_kw = r"\b(refund|refused|cancelled|cancellation)\b"
baseline_rate = df_neg['content'].str.lower().str.contains(refund_kw).mean()
print(f"▶️ Baseline 환불 언급율: {baseline_rate:.1%}")

# 예시로 이미 구해두신 솔루션별 부정 집단 유사도(sim_neg)를 딕셔너리로 준비
sim_neg = {
    "Trust Insights UI": 0.282,
    "Host Trust Score":   0.262,
    "Trust Insights Pro":  0.260,
    "Sponsored Listings":  0.238,
    "Localized Risk Index":0.058,
    # … 나머지 솔루션들도 동일하게 …
}

# 3) 예측 환불 언급 감소량 및 적용 후 언급율 계산
predictions = []
for sol, sim in sim_neg.items():
    drop = baseline_rate * sim          # 환불 이슈가 sim 만큼 해소된다고 가정
    new_rate = baseline_rate - drop
    predictions.append((sol, baseline_rate, drop, new_rate))

# 4) 결과를 보기 좋게 정리
pred_df = pd.DataFrame(predictions, columns=[
    "Solution", "Baseline Rate", "Predicted Drop", "New Rate"
])
pred_df["Baseline Rate"] = pred_df["Baseline Rate"].map("{:.1%}".format)
pred_df["Predicted Drop"] = pred_df["Predicted Drop"].map("{:.1%}".format)
pred_df["New Rate"]       = pred_df["New Rate"].map("{:.1%}".format)

print(pred_df.sort_values("Predicted Drop", ascending=False))
