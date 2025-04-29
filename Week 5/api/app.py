from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("irisModel.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    data = request.get_json()
    features = [
        float(data["sepal_length"]), 
        float(data["sepal_width"]),
        float(data["petal_length"]), 
        float(data["petal_width"])
    ]
    prediction = model.predict([features])[0]
    species = ["setosa", "versicolor", "virginica"]
    return jsonify({"prediction": species[prediction]})