from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

def trainSaveModel():
    # Load iris dataset to be trained on
    iris = load_iris()
    # Use all of the Iris dataset for training now that the model has been created
    X, y = iris.data, iris.target

    # Train model using RandomForestClassifier
    model = RandomForestClassifier(n_estimators = 100, random_state = 42)
    model.fit(X, y)

    # Save model and let user know it's saved
    joblib.dump(model, "api/irisModel.pkl")
    print("Production model saved to api/irisModel.pkl")

if __name__ == "__main__":
    trainSaveModel()