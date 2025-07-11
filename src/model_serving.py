
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = None

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json(force=True)
    df = pd.DataFrame(data)

    predictions = model.predict(df)
    return jsonify(predictions.tolist())

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None})

if __name__ == "__main__":
    # Load the trained model when the Flask app starts
    try:
        model = joblib.load("models/trained_model.pkl")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
    app.run(host="0.0.0.0", port=5000)


