from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("irisModel.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    data = request.get_json()
    features = pd.DataFrame([[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]], columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"])
    prediction = model.predict(features)[0]
    species = ["setosa", "versicolor", "virginica"]
    return jsonify({"prediction": species[prediction]})

if __name__ == "__main__":
    app.run(debug = True)