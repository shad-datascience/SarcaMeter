from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np
import os

app = Flask(__name__)

import pickle
import gzip

# Load the compressed Pickle file
with gzip.open('data.pkl.gz', 'rb') as f:
    data = pickle.load(f)

# print("Loaded Data:", data)
pipeline = data
# Load the pre-trained pipeline
# pipeline = joblib.load("model/feature_pipeline.pkl")

# Load the trained Keras model
model = keras.models.load_model("model/mlp_model.keras")


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None  # Initialize prediction result
    if request.method == "POST":
        text = request.form["text"]  # Get input text
        transformed_text = pipeline.transform([text]) #.toarray()  # Preprocess text

        # Get model prediction
        prediction = model.predict(transformed_text)
        if prediction>=0.5:
            prediction_result= "Sarcastic"
        else:
            prediction_result = "Not Sarcastic"
        # predicted_class = np.argmax(prediction, axis=1)[0]  # Get the class label

        print(prediction)  # Debugging: print the prediction array
        # return jsonify({"prediction": int(predicted_class)})
        
        # prediction_result = f"Predicted Class: {predicted_class}"

    # return render_template("index.html")  # Load frontend
    return render_template("index.html", prediction=prediction_result)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)