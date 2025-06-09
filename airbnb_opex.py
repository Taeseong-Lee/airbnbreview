import pandas as pd
import matplotlib.pyplot as plt

# 1) OPEX 불러오기
opex_data = {
    "quarter": [
        "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31",
        "2021-03-31", "2021-06-30", "2021-09-30", "2021-12-31",
        "2022-03-31", "2022-06-30", "2022-09-30", "2022-12-31",
        "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31",
        "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
        "2025-03-31"
    ],
    "opex_musd": [
        1167, 918, 924, 3959,
        1334, 1386, 1385, 1457,
        1514, 1735, 1681, 1667,
        1823, 1961, 1901, 2714,
        2041, 2251, 2207, 2050,
        2234
    ]
}
df_opex = pd.DataFrame(opex_data)
df_opex["quarter"] = pd.to_datetime(df_opex["quarter"])

# 2) 리뷰 데이터 불러오기 & 분기별 부정(1~2점) 비율 계산
df = pd.read_csv("airbnbview/airbnb_reviews_with_cluster.csv", parse_dates=["at"])
df["quarter"] = df["at"].dt.to_period("Q").dt.to_timestamp()
# ★ 부정 기준을 평점(score) ≤ 2 로 변경
df["is_negative"] = df["score"] <= 3

neg_share = (
    df.groupby("quarter")["is_negative"]
      .mean()         # 평균을 내면 비율이 된다(0~1)
      .mul(100)       # 퍼센트(0~100)
      .rename("neg_pct")
)

# 3) 두 데이터 병합
df_plot = df_opex.merge(neg_share, on="quarter", how="left").fillna(0)

# 4) 플롯
fig, ax1 = plt.subplots(figsize=(10, 6))

# — 왼쪽 축: OPEX
ax1.plot(df_plot["quarter"], df_plot["opex_musd"], "-o",
         color="tab:blue", label="OPEX (M USD)")
ax1.set_ylabel("Operating Expenses (M USD)", color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax1.ticklabel_format(axis="y", style="plain", useOffset=False)
ax1.set_ylim(0, df_plot["opex_musd"].max() * 1.1)

# — 오른쪽 축: 부정 리뷰 비율
ax2 = ax1.twinx()
ax2.plot(df_plot["quarter"], df_plot["neg_pct"], "-s",
         color="tab:orange", label="Negative Reviews (%)")
ax2.set_ylabel("Negative Reviews (%)", color="tab:orange")
ax2.tick_params(axis="y", labe