from flask import Flask, render_template, request, jsonify
import joblib
import os

# Initializes the Flask app
app = Flask(__name__, template_folder = "templates")

# Loads the model
modelPath = os.path.join(os.path.dirname(__file__), "irisModel.pkl")
model = joblib.load(modelPath)

# Route to render home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to make prediction
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
    return jsonify({"prediction": prediction[0]})

# Run the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host = "0.0.0.0", port = port)