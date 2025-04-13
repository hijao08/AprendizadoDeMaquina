import unittest
import pandas as pd
from data_preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        # Sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            "gas": ["N2O (t)", "CO2 (t)", None],
            "emissao": [100, None, 300],
            "other_column": ["abc", None, "def"]
        })
        self.expected_data = pd.DataFrame({
            "emissao": [100.0, 0.0, 300.0],
            "other_column_abc": [1, 0, 0],
            "other_column_def": [0, 0, 1],
            "other_column_desconhecido": [0, 1, 0],
            "gas_N2O (t)": [1, 0, 0],
            "gas_CO2 (t)": [0, 1, 0],
            "gas_desconhecido": [0, 0, 1]
        })

    def test_preprocess(self):
        # Initialize the DataPreprocessor with the sample data
        preprocessor = DataPreprocessor(self.sample_data)
        result = preprocessor.preprocess()

        # Assert the result matches the expected DataFrame
        pd.testing.assert_frame_equal(result.reset_index(drop=True), self.expected_data)

if __name__ == "__main__":
    unittest.main()