
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

def train_model(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

if __name__ == '__main__':
    # Example usage: Create dummy data and train model
    data = {'feature1': [1, 2, 3, 4, 5], 'feature2': [5, 4, 3, 2, 1], 'target': [0, 1, 0, 1, 0]}
    df = pd.DataFrame(data)
    model, X_test, y_test = train_model(df)
    print("Model trained successfully.")
    # Save the model for later use
    joblib.dump(model, 'trained_model.pkl')
    print("Model saved as trained_model.pkl")


