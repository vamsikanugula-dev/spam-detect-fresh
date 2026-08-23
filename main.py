from flask import Flask, request, jsonify
import joblib
import re
import os

app = Flask(__name__)

API_KEY = os.getenv("API_KEY")

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
report = joblib.load("report.pkl")


def preprocess(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' URL ', text)
    return text


@app.route("/", methods=["GET"])
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()

    html = html.replace("{{ prediction }}", "")
    return html


@app.route("/api/predict", methods=["POST"])
def api_predict():

    auth_header = request.headers.get("Authorization")

    if auth_header != f"Bearer {API_KEY}":
        return jsonify({"error": "Invalid API key"}), 401

    data = request.json

    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    message = data["message"]

    processed = preprocess(message)

    vector = vectorizer.transform([processed])

    pred = model.predict(vector)[0]

    if pred == 1:
        result = "⚠️ SPAM MESSAGE"
        color = "#ffffff"
        bgcolor = "#ff0000"
        glow = "#ff6666"
    else:
        result = "✅ SAFE MESSAGE"
        color = "#ffffff"
        bgcolor = "#00cc44"
        glow = "#66ff88"

    vector_words = vectorizer.get_feature_names_out()
    tokens = processed.split()

    suspicious_words = [
        w for w in tokens if w in vector_words
    ][:5] if pred == 1 else []

    spam_prob = float(model.predict_proba(vector)[0][1])

    accuracy = report["accuracy"]
    precision = report["1"]["precision"]
    recall = report["1"]["recall"]
    f1 = report["1"]["f1-score"]

    return jsonify({
        "prediction": result,
        "color": color,
        "bgcolor": bgcolor,
        "glow": glow,
        "spam_prob": spam_prob,
        "suspicious": suspicious_words,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)