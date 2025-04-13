import unittest
import pandas as pd
from io import StringIO
from data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Sample CSV data for testing
        self.sample_data = StringIO(
            """gas,emissao,other_column
            N2O (t),100,abc
            N2O (t),200,def
            CO2 (t),300,ghi
            N2O (t),,jkl"""
        )
        self.expected_data = pd.DataFrame({
            "emissao": [100.0, 200.0],
            "other_column": ["abc", "def"]
        })

    def test_load_data(self):
        # Mock the file path with StringIO object
        data_loader = DataLoader(self.sample_data)
        result = data_loader.load_data()

        # Assert the result matches the expected DataFrame
        pd.testing.assert_frame_equal(result.reset_index(drop=True), self.expected_data)

if __name__ == "__main__":
    unittest.main()