from flask import Flask, render_template, request
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import re

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

sentiment_model = pipeline(
    "sentiment-analysis",
    model="LiYuan/amazon-review-sentiment-analysis"
)

def analyze_sentiment_transformer(text):
    result = sentiment_model(text)[0]
    label = result['label']
    score = float(result['score'])
    return label, score

def tokenize(text: str) -> list[str]:
    if not isinstance(text, str):
        return []
    
    stopwords = {"the", "and", "is", "at", "to", "a", "of", "in", "it", "for", "on", "this", "that", "with", "i"}
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return words

def extract_keywords(reviews):
    freq = {}
    for review in reviews:
        words = tokenize(review)
        for word in words:
            if not word in freq:
                freq[word] = 1
            else:
                freq[word] += 1
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]

def generate_summary(df):
    return {
        "total_reviews": len(df),
        "positive_percent": (df['vader_label'] == "Positive").mean() * 100,
        "negative_percent": (df['vader_label'] == "Negative").mean() * 100,
        "neutral_percent": (df['vader_label'] == "Neutral").mean() * 100,
        "top_praise": extract_keywords(df[df['vader_label'] == "Positive"]['review'].tolist())[:3],
        "top_complaints": extract_keywords(df[df['vader_label'] == "Negative"]['review'].tolist())[:3],
        "overall_themes": extract_keywords(df['review'].tolist())
    }

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files['file']
    df = pd.read_csv(file)
    df['vader_score'] = df['review'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

    def vader_label(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"
        
    df['vader_label'] = df['vader_score'].apply(vader_label)
    df['transformer_result'] = df['review'].apply(lambda x: analyze_sentiment_transformer(str(x)))
    df['transformer_label'] = df['transformer_result'].apply(lambda x: x[0])
    df['transformer_confidence'] = df['transformer_result'].apply(lambda x: x[1])
    df = df.drop(columns=['transformer_result'])

    return df.to_html()

if __name__ == "__main__":
    app.run(debug=True)
