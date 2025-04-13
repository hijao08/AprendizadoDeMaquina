import unittest
from unittest.mock import MagicMock
from sklearn.metrics import mean_squared_error
from model_evaluator import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        # Mock test data
        self.X_test = [[1], [2], [3]]
        self.y_test = [1.5, 2.5, 3.5]
        self.model = MagicMock()
        self.model.predict = MagicMock(return_value=[1.4, 2.6, 3.4])
        self.evaluator = ModelEvaluator(self.X_test, self.y_test)

    def test_evaluate(self):
        # Test the evaluate method
        mse = self.evaluator.evaluate(self.model, "MockModel")
        expected_mse = mean_squared_error(self.y_test, [1.4, 2.6, 3.4])
        self.assertAlmostEqual(mse, expected_mse, places=2)

    def test_compare_models(self):
        # Test the compare_models method
        mse_xgb = 2.0
        mse_baseline = 4.0
        with self.assertLogs(level='INFO') as log:
            self.evaluator.compare_models(mse_xgb, mse_baseline)
        melhoria_percentual = ((mse_baseline - mse_xgb) / mse_baseline) * 100
        self.assertIn(f"O modelo XGBoost é {melhoria_percentual:.2f}% melhor que o modelo de linha de base.", log.output[0])

if __name__ == "__main__":
    unittest.main()