from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

class ModelTrainer:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def train_xgboost(self):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        xgb_model = XGBRegressor(random_state=42, verbosity=0)
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=1)
        grid_search.fit(self.X_train, self.y_train)

        return grid_search.best_estimator_

    def train_baseline(self):
        modelo_baseline = LinearRegression()
        modelo_baseline.fit(self.X_train, self.y_train)
        return modelo_baseline 