import pandas as pd
import unittest
from src.data_preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        data = {
            'feature1': [1, 2, None, 4, 5],
            'feature2': [5, None, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(data)
        processed_df = preprocess_data(df)

        # Check if missing values in numerical columns are filled
        self.assertFalse(processed_df['feature1'].isnull().any())
        self.assertFalse(processed_df['feature2'].isnull().any())

        # Check if the mean was correctly imputed (for feature1, mean of [1,2,4,5] is 3)
        self.assertEqual(processed_df.loc[2, 'feature1'], 3.0)

        # Check if the mean was correctly imputed (for feature2, mean of [5,3,2,1] is 2.75)
        self.assertEqual(processed_df.loc[1, 'feature2'], 2.75)

if __name__ == '__main__':
    unittest.main()


