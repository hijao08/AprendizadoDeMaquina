import pandas as pd

class DataPreprocessor:
    def __init__(self, df):
        self.df = df

    def preprocess(self, df_externo=None):
        """
        Aplica o preprocessamento nos dados.
        
        Args:
            df_externo: DataFrame externo opcional para aplicar o mesmo preprocessamento
                        utilizado no DataFrame original
                        
        Returns:
            DataFrame preprocessado
        """
        # Utiliza o dataframe externo se fornecido, caso contrário usa o dataframe original
        df = df_externo.copy() if df_externo is not None else self.df.copy()
            
        # Tratamento de valores nulos
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("desconhecido")
            else:
                df[col] = df[col].fillna(0)
        
        # Codifica colunas categóricas com one-hot
        df = pd.get_dummies(df)
        return df 