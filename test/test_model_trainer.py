import unittest
from unittest.mock import MagicMock, patch
from model_trainer import ModelTrainer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        # Mock training data
        self.X_train = [[1], [2], [3], [4]]
        self.y_train = [1.5, 2.5, 3.5, 4.5]
        self.trainer = ModelTrainer(self.X_train, self.y_train)

    @patch('model_trainer.GridSearchCV')
    def test_train_xgboost(self, mock_grid_search):
        # Mock GridSearchCV behavior
        mock_best_model = MagicMock(spec=XGBRegressor)
        mock_grid_search.return_value.fit.return_value = None
        mock_grid_search.return_value.best_estimator_ = mock_best_model

        best_model = self.trainer.train_xgboost()

        # Assertions
        mock_grid_search.assert_called_once()
        mock_grid_search.return_value.fit.assert_called_once_with(self.X_train, self.y_train)
        self.assertEqual(best_model, mock_best_model)

    @patch('model_trainer.LinearRegression')
    def test_train_baseline(self, mock_linear_regression):
        # Mock LinearRegression behavior
        mock_model = MagicMock(spec=LinearRegression)
        mock_linear_regression.return_value = mock_model

        baseline_model = self.trainer.train_baseline()

        # Assertions
        mock_linear_regression.assert_called_once()
        mock_model.fit.assert_called_once_with(self.X_train, self.y_train)
        self.assertEqual(baseline_model, mock_model)

if __name__ == "__main__":
    unittest.main()