import pandas as pd
import statsmodels.api as sm

# ... (1~5번까지는 이전과 동일하게 df를 만드셨다고 가정)

# 모델 학습 (feature_on 제외)
X = sm.add_constant(df[["neg_pct"]])  # --> columns = ['const','neg_pct']
y = df["opex_musd"]
model = sm.OLS(y, X).fit()
print(model.summary())

# 향후 3분기만들기
last_q = df["quarter"].iloc[-1]
last_neg = float(df.loc[df["quarter"] == last_q, "neg_pct"])

future_q = [last_q + i for i in range(1, 4)]
future_df = pd.DataFrame({
    "quarter": future_q,
    "neg_pct": [last_neg]*3
})

# --- 핵심 수정 ---
# 1) add_constant 로 상수항을 넣고
future_X = sm.add_constant(future_df[["neg_pct"]], has_constant="add")
# 2) 모델이 학습할 때의 파라미터 순서(model.params.index)와 똑같이 재정렬
future_X = future_X[model.params.index]

print("예측용 exog 컬럼:", future_X.columns.tolist())
# 반드시 ['const','neg_pct'] 순서여야 합니다.

# 3) 이제 예측
future_df["pred_opex_musd"] = model.predict(future_X)

print("\n예측된 향후 3개 분기 OPEX:")
print(future_df[["quarter", "pred_opex_musd"]])
