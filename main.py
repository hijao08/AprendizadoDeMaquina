import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

# -------------------------------
# 1. Carregar dados e filtrar N2O
# -------------------------------
df = pd.read_csv("./data/input/br_seeg_emissoes_brasil.csv")

# Filtra apenas registros com gás N2O
df_n2o = df[df["gas"] == "N2O (t)"].copy()

# Remove linhas com emissão nula (NaN)
df_n2o = df_n2o[df_n2o["emissao"].notna()]

# Remove a coluna 'gas' após o filtro
df_n2o.drop(columns=["gas"], inplace=True)

# Garante a existência da pasta de saída
os.makedirs("./data/output", exist_ok=True)

# -----------------------------
# 2. Tratamento de valores nulos
# -----------------------------
for col in df_n2o.columns:
    if df_n2o[col].dtype == "object":
        df_n2o[col] = df_n2o[col].fillna("desconhecido")
    else:
        df_n2o[col] = df_n2o[col].fillna(0)

# -----------------------------
# 3. Análise exploratória
# -----------------------------
print("\n📌 Dimensões do dataframe filtrado (apenas N2O):")
print(df_n2o.shape)
print("\n🧾 Primeiras linhas:")
print(df_n2o.head())
print("\n🔍 Tipos de dados:")
print(df_n2o.dtypes)
print("\n❓ Valores nulos:")
print(df_n2o.isnull().sum())
print("\n📊 Estatísticas da emissão:")
print(df_n2o["emissao"].describe())

# Gráfico: distribuição da emissão
plt.figure(figsize=(10, 5))
sns.histplot(df_n2o["emissao"], bins=30, kde=True)
plt.title("Distribuição dos valores de emissão (N2O)")
plt.xlabel("Emissão (toneladas)")
plt.ylabel("Frequência")
plt.grid(True)
plt.tight_layout()
plt.savefig("./data/output/distribuicao_emissao.png")
plt.close()

# Frequência das categorias em nivel_1
if "nivel_1" in df_n2o.columns:
    plt.figure(figsize=(10, 4))
    sns.countplot(data=df_n2o, x="nivel_1", order=df_n2o["nivel_1"].value_counts().index)
    plt.xticks(rotation=45)
    plt.title("Frequência por categoria - nivel_1")
    plt.tight_layout()
    plt.savefig("./data/output/frequencia_nivel1.png")
    plt.close()

# Correlação numérica
plt.figure(figsize=(8, 6))
sns.heatmap(df_n2o.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação (variáveis numéricas)")
plt.tight_layout()
plt.savefig("./data/output/correlacao_numerica.png")
plt.close()

# -----------------------------------
# 4. Preparação dos dados para treino
# -----------------------------------
X = df_n2o.drop(columns=["emissao"])
y = df_n2o["emissao"]

# Codifica colunas categóricas com one-hot
X = pd.get_dummies(X)

# Normaliza os dados numéricos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separação treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ----------------------
# 5. Treinamento com XGBoost
# ----------------------
modelo = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
modelo.fit(X_train, y_train)

# ----------------------
# 5. Treinamento com Regressão Linear (Linha de Base)
# ----------------------
modelo_baseline = LinearRegression()
modelo_baseline.fit(X_train, y_train)

# -------------------------
# 6. Previsão e avaliação
# -------------------------
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\n✅ MSE (erro quadrático médio) - XGBoost: {mse:.2f}")

# Previsão e avaliação do modelo de linha de base
y_pred_baseline = modelo_baseline.predict(X_test)
mse_baseline = mean_squared_error(y_test, y_pred_baseline)
print(f"\n✅ MSE (erro quadrático médio) - Regressão Linear (Linha de Base): {mse_baseline:.2f}")

# Comparação de melhoria
melhoria_percentual = ((mse_baseline - mse) / mse_baseline) * 100
print(f"\n📈 A linha de base é {melhoria_percentual:.2f}% melhor que o modelo XGBoost.")

# --------------------------
# 7. Gerar CSV de resultado
# --------------------------
df_resultado = pd.DataFrame(X_test, columns=X.columns)
df_resultado["emissao_real"] = y_test.values
df_resultado["emissao_prevista"] = y_pred.round(2)

# Reorganiza colunas colocando 'ano' primeiro se existir
colunas = df_resultado.columns.tolist()
colunas_ordenadas = [c for c in ["ano"] if c in colunas] + \
                    [c for c in colunas if c not in ["ano", "emissao_real", "emissao_prevista"]] + \
                    ["emissao_real", "emissao_prevista"]
df_resultado = df_resultado[colunas_ordenadas]

df_resultado.to_csv("./data/output/resultado_previsto.csv", index=False)
print("📁 Arquivo 'resultado_previsto.csv' salvo com sucesso!")

# --------------------------
# 8. Gráfico real vs previsto
# --------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Valor real")
plt.ylabel("Valor previsto")
plt.title("Comparação: Real vs Previsto (XGBoost)")
plt.grid(True)
plt.tight_layout()
plt.savefig("./data/output/real_vs_previsto.png")
plt.close()

# -------------------------------
# 9. Feature Importance (Gráfico)
# -------------------------------
plt.figure(figsize=(12, 6))
importances = modelo.feature_importances_
indices = np.argsort(importances)[-20:]  # Top 20 features
plt.barh(np.array(X.columns)[indices], importances[indices])
plt.title("Importância das Variáveis - XGBoost")
plt.xlabel("Score")
plt.tight_layout()
plt.savefig("./data/output/importancia_variaveis.png")
plt.close()
