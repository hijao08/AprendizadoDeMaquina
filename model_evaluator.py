from sklearn.metrics import mean_squared_error

class ModelEvaluator:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self, model, model_name):
        y_pred = model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        print(f"\n✅ MSE (erro quadrático médio) - {model_name}: {mse:.2f}")
        return mse

    def compare_models(self, mse_xgb, mse_baseline):
        melhoria_percentual = ((mse_baseline - mse_xgb) / mse_baseline) * 100
        print(f"\n📈 O modelo XGBoost é {melhoria_percentual:.2f}% melhor que o modelo de linha de base.") 