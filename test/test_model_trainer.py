import pytest
from unittest.mock import MagicMock, patch
from src.model_trainer import ModelTrainer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

@pytest.fixture
def trainer_data():
    # Mock training data
    X_train = [[1], [2], [3], [4]]
    y_train = [1.5, 2.5, 3.5, 4.5]
    return X_train, y_train

@pytest.fixture
def model_trainer(trainer_data):
    X_train, y_train = trainer_data
    return ModelTrainer(X_train, y_train)

@patch('src.model_trainer.GridSearchCV')
def test_train_xgboost(mock_grid_search, model_trainer, trainer_data):
    X_train, y_train = trainer_data
    # Mock GridSearchCV behavior
    mock_best_model = MagicMock(spec=XGBRegressor)
    mock_grid_search.return_value.fit.return_value = None
    mock_grid_search.return_value.best_estimator_ = mock_best_model

    best_model = model_trainer.train_xgboost()

    # Assertions
    mock_grid_search.assert_called_once()
    mock_grid_search.return_value.fit.assert_called_once_with(X_train, y_train)
    assert best_model == mock_best_model

@patch('src.model_trainer.LinearRegression')
def test_train_baseline(mock_linear_regression, model_trainer, trainer_data):
    X_train, y_train = trainer_data
    # Mock LinearRegression behavior
    mock_model = MagicMock(spec=LinearRegression)
    mock_linear_regression.return_value = mock_model

    baseline_model = model_trainer.train_baseline()

    # Assertions
    mock_linear_regression.assert_called_once()
    mock_model.fit.assert_called_once_with(X_train, y_train)
    assert baseline_model == mock_model

def test_example():
    assert True