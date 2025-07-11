
import unittest
import requests
import json
import time
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

class TestModelServing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure the models directory exists
        os.makedirs("../models", exist_ok=True)

        # Create a dummy model for testing
        dummy_model = LogisticRegression()
        dummy_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        dummy_target = pd.Series([0, 1, 0, 1, 0])
        dummy_model.fit(dummy_data, dummy_target)
        joblib.dump(dummy_model, "../models/trained_model.pkl")
        print("Dummy model created for testing.")

        # Start the serving container (assuming it's built and available)
        # This part would typically be handled by a separate script or CI/CD
        # For this test, we'll assume the deploy_pipeline.sh has been run
        # and the service is up and running on port 5000.
        # We'll just wait for it to be ready.
        print("Waiting for model serving API to be ready...")
        for _ in range(30): # Wait up to 30 seconds
            try:
                response = requests.get("http://localhost:5000/health")
                if response.status_code == 200 and response.json().get("status") == "healthy":
                    print("Model serving API is ready.")
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
        else:
            raise Exception("Model serving API did not become ready in time.")

    @classmethod
    def tearDownClass(cls):
        # Clean up the dummy model
        os.remove("../models/trained_model.pkl")
        print("Dummy model removed.")
        # Optionally, stop the Docker container if it was started by this test
        # For now, we assume it's managed externally or will be stopped manually.

    def test_predict_endpoint(self):
        test_data = {
            "feature1": [6, 7],
            "feature2": [0, 1]
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post("http://localhost:5000/predict", data=json.dumps(test_data), headers=headers)
        self.assertEqual(response.status_code, 200)
        predictions = response.json()
        self.assertIsInstance(predictions, list)
        self.assertEqual(len(predictions), 2)

    def test_health_endpoint(self):
        response = requests.get("http://localhost:5000/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
        self.assertTrue(response.json()["model_loaded"])

if __name__ == '__main__':
    unittest.main()


