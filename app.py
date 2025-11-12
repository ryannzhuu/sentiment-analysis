from flask import Flask, render_template, request
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()

def tokenize(text):
    stopwords = {"the", "and", "is", "at", "to", "a", "of", "in", "it", "for", "on", "this", "that", "with", "i"}
    text = text.lower()
    for char in text:
        if not char.isalnum() and not char.isspace():
            text = text.replace(char, "")
    text = text.split()
    for word in text:
        if word in stopwords:
            text.remove(word)
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

    # Apply VADER sentiment scoring to each review
    df['sentiment_score'] = df['review'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

    # Convert numeric score → label
    def label(score):
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    df['sentiment_label'] = df['sentiment_score'].apply(label)

    return df.to_html()

if __name__ == "__main__":
    app.run(debug=True)
