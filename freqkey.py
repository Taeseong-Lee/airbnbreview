# keyword_top_positive.py
import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt

# ── 1. 데이터 로드 ──────────────────────────────
df  = pd.read_csv("airbnbview/cluster_0_reviews.csv")
texts = df["content"].dropna().astype(str).tolist()

# ── 2. SpaCy 모델(영어) 로딩 ───────────────────
#  ➜ 처음 실행 시 python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # 속도 ↑

# ── 3. 전처리 & 표제어 추출 ────────────────────
tokens = []
for doc in nlp.pipe(texts, batch_size=500):
    for tok in doc:
        # keep nouns / adjectives, length ≥3, delete stopwords & punctuation
        if tok.is_stop or tok.is_punct:
            continue
        if tok.pos_ not in ("NOUN", "ADJ"):
            continue
        lemma = tok.lemma_.lower()
        if len(lemma) < 3:
            continue
        tokens.append(lemma)

# ── 4. 빈도 집계 & 상위 키워드 선택 ─────────────
freq = Counter(tokens)
# 문서 빈도 기준 필터 (at least 5 occurrences)
cleaned = {k: v for k, v in freq.items() if v >= 5}
top_n = 15
top_terms = Counter(cleaned).most_common(top_n)

# ── 5. 결과 출력 & 시각화(optional) ──────────────
print("🟢 Top keywords in Positive Cluster")
for word, cnt in top_terms:
    print(f"{word:<12} {cnt}")

# 막대그래프
if top_terms:
    words, counts = zip(*top_terms)
    plt.figure(figsize=(8,5))
    plt.barh(words[::-1], counts[::-1], color="#4C9AFF")
    plt.title("Top Keywords (Positive Cluster)")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.show()