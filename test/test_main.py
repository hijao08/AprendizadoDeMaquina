import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from main import DataLoader, DataPreprocessor, ExploratoryAnalysis, ModelTrainer, ModelEvaluator

class TestMain(unittest.TestCase):
    @patch("main.DataLoader")
    def test_data_loading(self, MockDataLoader):
        # Mock the DataLoader
        mock_loader = MockDataLoader.return_value
        mock_loader.load_data.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        
        data_loader = DataLoader("./data/input/br_seeg_emissoes_brasil.csv")
        df = data_loader.load_data()
        
        mock_loader.load_data.assert_called_once()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))

    @patch("main.DataPreprocessor")
    def test_data_preprocessing(self, MockDataPreprocessor):
        # Mock the DataPreprocessor
        mock_preprocessor = MockDataPreprocessor.return_value
        mock_preprocessor.preprocess.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        
        preprocessor = DataPreprocessor(pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}))
        df = preprocessor.preprocess()
        
        mock_preprocessor.preprocess.assert_called_once()
        self.assertIsInstance(df, pd.DataFrame)

    @patch("main.ModelTrainer")
    def test_model_training(self, MockModelTrainer):
        # Mock the ModelTrainer
        mock_trainer = MockModelTrainer.return_value
        mock_trainer.train_xgboost.return_value = GradientBoostingRegressor()
        mock_trainer.train_baseline.return_value = GradientBoostingRegressor()
        
        trainer = ModelTrainer(np.array([[1, 2], [3, 4]]), np.array([1, 2]))
        xgb_model = trainer.train_xgboost()
        baseline_model = trainer.train_baseline()
        
        mock_trainer.train_xgboost.assert_called_once()
        mock_trainer.train_baseline.assert_called_once()
        self.assertIsInstance(xgb_model, GradientBoostingRegressor)
        self.assertIsInstance(baseline_model, GradientBoostingRegressor)

    @patch("main.ModelEvaluator")
    def test_model_evaluation(self, MockModelEvaluator):
        # Mock the ModelEvaluator
        mock_evaluator = MockModelEvaluator.return_value
        mock_evaluator.evaluate.return_value = 0.5
        
        evaluator = ModelEvaluator(np.array([[1, 2], [3, 4]]), np.array([1, 2]))
        mse = evaluator.evaluate(GradientBoostingRegressor(), "XGBoost")
        
        mock_evaluator.evaluate.assert_called_once()
        self.assertEqual(mse, 0.5)

if __name__ == "__main__":
    unittest.main()