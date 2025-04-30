from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model = joblib.load("irisModel.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    data = request.get_json()
    features = [[
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]]
    prediction = model.predict(features)[0]
    species = ["setosa", "versicolor", "virginica"]
    return jsonify({"prediction": species[prediction]})

if __name__ == "__main__":
    app.run(debug = True)