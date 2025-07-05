import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from transformers import pipeline

# 데이터 읽기
df = pd.read_csv("dream_data.csv")

st.title("꿈 데이터 워드클라우드 & 감정 분석")

# 꿈 내용 합치기
text = " ".join(df["꿈 내용"])

# 띄어쓰기 기준으로 단어 분리
words = text.split()

# 단어 빈도 계산
word_counts = Counter(words)

# 워드클라우드 생성 (한글 폰트 경로는 환경에 맞게 수정)
wc = WordCloud(font_path='NanumGothic.ttf', background_color='white', width=800, height=400)
cloud = wc.generate_from_frequencies(word_counts)

st.subheader("꿈에서 많이 나온 단어들 (워드클라우드)")
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(cloud, interpolation='bilinear')
ax.axis('off')
st.pyplot(fig)

# 감정 분석
st.subheader("꿈 내용 감정 분석")
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

results = []
for sentence in df["꿈 내용"]:
    result = classifier(sentence)
    results.append(result[0]['label'])

df["감정 분석 결과"] = results
st.dataframe(df)
