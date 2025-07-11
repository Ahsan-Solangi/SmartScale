
import pandas as pd

def preprocess_data(df):
    # Simple preprocessing: fill missing values with mean for numerical columns
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = df[col].fillna(df[col].mean())
    return df

if __name__ == '__main__':
    # Example usage: Create dummy data and preprocess
    data = {'feature1': [1, 2, None, 4, 5], 'feature2': [5, None, 3, 2, 1], 'target': [0, 1, 0, 1, 0]}
    df = pd.DataFrame(data)
    print("Original DataFrame:\n", df)
    processed_df = preprocess_data(df)
    print("\nProcessed DataFrame:\n", processed_df)


