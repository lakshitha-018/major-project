import streamlit as st
from transformers import pipeline

# Load model
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

pipe = load_model()

# Map detailed sentiment to 3 classes
def map_sentiment(label):
    label = label.lower()
    if "very negative" in label or "negative" in label:
        return "negative"
    elif "neutral" in label:
        return "neutral"
    elif "positive" in label or "very positive" in label:
        return "positive"
    else:
        return "unknown"

# UI
st.title("🌍  Sentiment Classifier")


text_input = st.text_area("Enter text to analyze", placeholder="Write your sentence here...")

if st.button("Analyze Sentiment"):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        prediction = pipe(text_input)[0]
        simplified_sentiment = prediction["label"]

        st.subheader("📊 Model Output")
        st.json(prediction)

        st.subheader("🧠 Simplified Sentiment")
        st.success(f"**{simplified_sentiment.upper()}**")
