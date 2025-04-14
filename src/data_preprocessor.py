import pandas as pd

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess(self):
        # Tratamento de valores nulos
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].fillna("desconhecido")
            else:
                self.df[col] = self.df[col].fillna(0)
        
        # Codifica colunas categóricas com one-hot
        self.df = pd.get_dummies(self.df)
        return self.df 