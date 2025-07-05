import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

st.title("Dream Data Wordcloud, Sentiment & Emoji Representation")

# 데이터 읽기
df = pd.read_csv("dream_data.csv")

# ----- 워드클라우드 -----
text = " ".join(df["dream"].dropna())
words = text.split()
word_counts = Counter(words)

wc = WordCloud(background_color='white', width=800, height=400)
cloud = wc.generate_from_frequencies(word_counts)

st.subheader("Common words in dreams (Wordcloud)")
fig, ax = plt.subplots(figsize=(10,5))
ax.imshow(cloud, interpolation='bilinear')
ax.axis("off")
st.pyplot(fig)

# ----- 간단 감정 분석 -----
positive_words = ['happy', 'good', 'free', 'love', 'peace', 'calm']
negative_words = ['lost', 'fear', 'scared', 'stress', 'late', 'dark']

def sentiment_check(text):
    text = str(text).lower()
    if any(word in text for word in positive_words):
        return "Positive"
    elif any(word in text for word in negative_words):
        return "Negative"
    else:
        return "Error"

df["Sentiment"] = df["dream"].apply(sentiment_check)

# ----- 이모지 변환 -----
def dream_to_emoji(text):
    text = str(text).lower()
    emojis = []
    if "sea" in text or "ocean" in text or "water" in text:
        emojis.append("🌊")
    if "fly" in text or "sky" in text:
        emojis.append("🕊️")
    if "cat" in text:
        emojis.append("🐈‍⬛")
    if "dog" in text:
        emojis.append("🐕")
    if "rain" in text:
        emojis.append("🌧️")
    if "forest" in text or "tree" in text:
        emojis.append("🌳")
    if "school" in text:
        emojis.append("🏫")
    if "test" in text or "exam" in text:
        emojis.append("📝")
    if "fall" in text or "drop" in text:
        emojis.append("⬇️")
    if "chase" in text or "run" in text:
        emojis.append("🏃")
    
    return " ".join(emojis) if emojis else "❓"

df["Emoji"] = df["dream"].apply(dream_to_emoji)

# ----- 결과 표시 -----
st.subheader("Dream Sentiment & Emoji")
st.dataframe(df[["dream", "Sentiment", "Emoji"]])
