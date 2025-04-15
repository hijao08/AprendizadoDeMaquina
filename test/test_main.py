import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from src.model_evaluator import ModelEvaluator
import sys
from src.model_trainer import ModelTrainer as NewModelTrainer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

sys.modules['src.main'] = MagicMock()

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "gas": ["N2O (t)", "CO2 (t)"], 
        "emissao": [100, 200],
        "other_column": ["abc", "def"]
    })

@pytest.fixture
def model_trainer():
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([1, 2])
    return NewModelTrainer(X_train, y_train)

@patch("pandas.read_csv")
def test_data_loading(mock_read_csv, sample_data):
    from src.data_loader import DataLoader
    mock_read_csv.return_value = sample_data
    
    data_loader = DataLoader("./data/input/br_seeg_emissoes_brasil.csv")
    df = data_loader.load_data()
    
    mock_read_csv.assert_called_once()
    
    assert "gas" not in df.columns
    assert "emissao" in df.columns
    assert df.shape[0] == 1 

def test_data_preprocessing():
    from src.data_preprocessor import DataPreprocessor
    test_df = pd.DataFrame({
        "gas": ["N2O (t)", "CO2 (t)"],
        "emissao": [100, None],
        "other_column": ["abc", None]
    })
    
    preprocessor = DataPreprocessor(test_df)
    result_df = preprocessor.preprocess()
    
    assert result_df["emissao"].isna().sum() == 0
    
    assert len(result_df.columns) > len(test_df.columns)
    assert "gas_N2O (t)" in result_df.columns or "gas_N2O_(t)" in result_df.columns

@pytest.fixture
def training_data():
    X_train = [[1], [2], [3], [4]]
    y_train = [1.5, 2.5, 3.5, 4.5]
    return X_train, y_train

@patch('src.model_trainer.GridSearchCV')
def test_train_xgboost(mock_grid_search, training_data):
    X_train, y_train = training_data
    trainer = NewModelTrainer(X_train, y_train)

    mock_best_model = MagicMock(spec=XGBRegressor)
    mock_grid_search.return_value.fit.return_value = None
    mock_grid_search.return_value.best_estimator_ = mock_best_model

    best_model = trainer.train_xgboost()

    mock_grid_search.assert_called_once()
    mock_grid_search.return_value.fit.assert_called_once_with(X_train, y_train)
    assert best_model == mock_best_model

@patch('src.model_trainer.LinearRegression')
def test_train_baseline(mock_linear_regression, training_data):
    X_train, y_train = training_data
    trainer = NewModelTrainer(X_train, y_train)

    # Mock inearRegression behavior
    mock_model = MagicMock(spec=LinearRegression)
    mock_linear_regression.return_value = mock_model

    baseline_model = trainer.train_baseline()

    mock_linear_regression.assert_called_once()
    mock_model.fit.assert_called_once_with(X_train, y_train)
    assert baseline_model == mock_model

def test_model_evaluation():
    X_test = np.array([[1, 2], [3, 4]])
    y_test = np.array([1.5, 2.5])
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1.6, 2.4])
    
    evaluator = ModelEvaluator(X_test, y_test)
    mse = evaluator.evaluate(mock_model, "MockModel")
    
    mock_model.predict.assert_called_once_with(X_test)
    
    expected_mse = ((1.6 - 1.5)**2 + (2.4 - 2.5)**2) / 2
    assert round(mse, 4) == round(expected_mse, 4)