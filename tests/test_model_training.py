
import pandas as pd
import unittest
from sklearn.linear_model import LogisticRegression
from src.model_training import train_model

class TestModelTraining(unittest.TestCase):
    def test_train_model(self):
        data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        model, X_test, y_test = train_model(df)

        # Check if the returned model is an instance of LogisticRegression
        self.assertIsInstance(model, LogisticRegression)

        # Check if X_test and y_test are pandas DataFrames/Series
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.Series)

        # Check if the model has been fitted (has coef_ attribute)
        self.assertTrue(hasattr(model, 'coef_'))

if __name__ == '__main__':
    unittest.main()


