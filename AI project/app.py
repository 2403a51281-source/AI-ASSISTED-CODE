from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form["news"]
    vectorized = vectorizer.transform([news_text])
    prediction = model.predict(vectorized)[0]

    result = "FAKE NEWS ❌" if prediction == "fake" else "REAL NEWS ✔️"
    color = "red" if prediction == "fake" else "green"

    return render_template("index.html", prediction=result, color=color, input_text=news_text)

if __name__ == "__main__":
    app.run(debug=True)
