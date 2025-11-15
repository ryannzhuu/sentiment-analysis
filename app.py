from flask import Flask, render_template, request
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

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

def tokenize(text):
    stopwords = {"the", "and", "is", "at", "to", "a", "of", "in", "it", "for", "on", "this", "that", "with", "i"}
    text = text.lower()
    for char in text:
        if not char.isalnum() and not char.isspace():
            text = text.replace(char, "")
    text = text.split()
    for word in text:
        if word in stopwords:
            text = [word for word in text if word not in stopwords]
    return text

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
