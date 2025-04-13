import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path)
        df_n2o = df[df["gas"] == "N2O (t)"].copy()
        df_n2o = df_n2o[df_n2o["emissao"].notna()]
        df_n2o.drop(columns=["gas"], inplace=True)
        return df_n2o 