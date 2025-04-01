# Bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Métricas de avaliação
from sklearn.metrics import mean_squared_error, r2_score

# Carregar os dados
df = pd.read_csv('./br_seeg_emissoes_brasil.csv')

# Filtrar apenas dados de N2O
df_n2o = df[df['gas'] == 'Óxido Nitroso (N2O)']

# Verificar dados faltantes
print("Dados faltantes:\n", df_n2o.isnull().sum())

# Estatísticas básicas
print("Estatísticas descritivas:\n", df_n2o.describe())

# Verificar tipos de dados
print("\nTipos de dados:\n", df_n2o.dtypes)