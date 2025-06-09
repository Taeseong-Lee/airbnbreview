import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

# ── 1. 데이터 준비 ──
# 총 리뷰 수 (예: 1000건이라고 가정)
N = 1000

# 베이스라인 환불 언급 건수
baseline_rate = 0.06
baseline_count = int(baseline_rate * N)

# 솔루션별 예측 언급율 (예: Trust Insights UI가 1.7% 언급율로 떨어졌다고 가정)
solutions = {
    "Trust Insights UI": 0.017,
    "Host Trust Score":    0.016,
    "Trust Insights Pro":  0.016,
    "Sponsored Listings":  0.014,
    "Localized Risk Index":0.003,
}

# ── 2. Z-검정 & p-value 계산 ──
results = []
for name, new_rate in solutions.items():
    new_count = int(new_rate * N)
    # 두 집단(베이스라인 vs 솔루션)에서 '환불 언급'이 발생한 건수, 전체 리뷰 수
    count = np.array([baseline_count, new_count])
    nobs  = np.array([N, N])
    stat, pval = proportions_ztest(count, nobs, alternative='larger')  
    # alternative='larger' 는 솔루션 환불율 < 베이스라인인지 검정
    results.append((name, baseline_rate, new_rate, pval))

# ── 3. 결과 출력 ──
df = pd.DataFrame(results, columns=["Solution", "Baseline Rate", "Predicted Rate", "p-value"])
print(df.to_string(index=False))
