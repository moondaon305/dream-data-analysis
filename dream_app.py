import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

st.title("Dream Data Wordcloud & Simple Sentiment Analysis")

# 데이터 읽기 (컬럼명이 'dream'이어야 함)
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

# 간단 감정 분석용 단어 리스트
positive_words = ['happy', 'good', 'free', 'love', 'peace', 'calm']
negative_words = ['lost', 'fear', 'scared', 'stress', 'late', 'dark']

def sentiment_check(text):
    text = str(text).lower()
    if any(word in text for word in positive_words):
        return "Positive"
    elif any(word in text for word in negative_words):
        return "Negative"
    else:
        return "Error"  # 감정 판별 불가

df["Sentiment"] = df["dream"].apply(sentiment_check)

st.subheader("Simple Sentiment Analysis Result")
st.dataframe(df)
