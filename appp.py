import numpy as np
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model1.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the feature values from the request
    features = request.json["Data"]
    # Make predictions
    prediction = list(model.predict(np.array(features)))

    # Return the predicted class
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)