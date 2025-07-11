
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    # This part would typically be run after model_training.py
    # For demonstration, let's assume we have a dummy model and test data
    from sklearn.linear_model import LogisticRegression
    import pandas as pd

    # Dummy data for evaluation
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    X = df.drop("target", axis=1)
    y = df["target"]

    # Create a dummy model and make predictions
    model = LogisticRegression()
    model.fit(X, y) # Train on full data for simplicity in this example
    X_test = X
    y_test = y

    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")


