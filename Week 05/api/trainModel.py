from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def trainSaveModel():
    # Load iris dataset to be trained on
    iris = load_iris()
    # Use all of the Iris dataset for training now that the model has been created
    X, y = iris.data, iris.target

    # Train model using RandomForestClassifier
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(X, y)

    # Save model and let user know it's saved
    modelPath = os.path.join(os.path.dirname(__file__), "irisModel.pkl")
    joblib.dump(model, modelPath)
    print("Production model saved to {modelPath}")

if __name__ == "__main__":
    trainSaveModel()