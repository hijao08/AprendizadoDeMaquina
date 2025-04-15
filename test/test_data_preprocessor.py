import pytest
import pandas as pd
import numpy as np
from src.data_preprocessor import DataPreprocessor

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "gas": ["N2O (t)", "CO2 (t)", None],
        "emissao": [100, None, 300],
        "other_column": ["abc", None, "def"]
    })

@pytest.fixture
def expected_data():
    return pd.DataFrame({
        "emissao": [100.0, 0.0, 300.0],
        "other_column_abc": [1, 0, 0],
        "other_column_def": [0, 0, 1],
        "other_column_desconhecido": [0, 1, 0],
        "gas_N2O (t)": [1, 0, 0],
        "gas_CO2 (t)": [0, 1, 0],
        "gas_desconhecido": [0, 0, 1]
    })

def test_preprocess(sample_data, expected_data):
    preprocessor = DataPreprocessor(sample_data)
    result = preprocessor.preprocess()
    
    expected_columns = set(expected_data.columns)
    result_columns = set(result.columns)
    assert expected_columns.issubset(result_columns)
    
    assert (result["emissao"] == expected_data["emissao"]).all()
    
    for col in expected_data.columns:
        if col != "emissao":
            assert col in result.columns
            
            result_values = result[col].astype(int).values
            expected_values = expected_data[col].values
            
            np.testing.assert_array_equal(result_values, expected_values)