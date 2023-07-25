import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("pipe.pkl", "rb"))


@flask_app.route("/predict", methods = ["POST"])
def predict():
          json_=request.json["Text"]
          prediction = list(model.predict(np.array(json_)))
          return jsonify({"Prediction of the movie review" : prediction})

if __name__ == "__main__":
    flask_app.run(debug=True)