import pandas as pd

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess(self, df_externo=None):
        df = df_externo.copy() if df_externo is not None else self.df.copy()
            
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("desconhecido")
            else:
                df[col] = df[col].fillna(0)
        
        df = pd.get_dummies(df)
        return df 