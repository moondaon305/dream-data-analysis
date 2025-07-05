import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from transformers import pipeline

st.title("Dream Data Wordcloud & Sentiment Analysis")

# 데이터 읽기 (컬럼명이 'dream'으로 되어 있어야 함)
df = pd.read_csv("dream_data.csv")

# 꿈 내용 합치기
text = " ".join(df["dream"].dropna())

# 단어 빈도 계산
words = text.split()
word_counts = Counter(words)

# 워드클라우드 생성
wc = WordCloud(background_color='white', width=800, height=400)
cloud = wc.generate_from_frequencies(word_counts)

st.subheader("Common words in dreams (Wordcloud)")
fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(cloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# 감정 분석 (CPU 강제 지정, 모델명 명시)
st.subheader("Sentiment Analysis of Dreams")
classifier = pipeline(
    "sentiment-analysis", 
    model="distilbert-base-uncased-finetuned-sst-2-english", 
    device=-1
)

results = []
for sentence in df["dream"]:
    if not isinstance(sentence, str) or sentence.strip() == "":
        results.append("No Text")
        continue
    sentence = sentence[:256]
    try:
        result = classifier(sentence)
        results.append(result[0]['label'])
    except Exception:
        results.append("Error")

df["Sentiment"] = results
st.dataframe(df)
