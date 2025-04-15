import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import sys
import joblib

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from exploratory_analysis import ExploratoryAnalysis
from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator

def processar_arquivo_teste(caminho_teste, melhor_modelo, preprocessador, escalar):
    """
    Processa o arquivo de teste e adiciona as previsões do modelo.
    
    Args:
        caminho_teste: Caminho para o arquivo de teste
        melhor_modelo: Modelo treinado para fazer as previsões
        preprocessador: Preprocessador para padronizar os dados
        escalar: Escalar usado para normalizar os dados
    """
    try:
        # Carregar arquivo de teste
        df_teste = pd.read_csv(caminho_teste)
        
        print(f"📊 Arquivo de teste carregado: {caminho_teste}")
        print(f"   Dimensões: {df_teste.shape}")
        
        # Aplicar o mesmo preprocessamento usado nos dados de treinamento
        df_teste_processado = preprocessador.preprocess(df_teste)
        
        # Preparar para previsão
        X_teste = df_teste_processado.drop(columns=["emissao"]) if "emissao" in df_teste_processado.columns else df_teste_processado
        
        # Aplicar a normalização
        X_teste_escalado = escalar.transform(X_teste)
        
        # Fazer previsão
        previsoes = melhor_modelo.predict(X_teste_escalado).round(2)
        
        # Adicionar previsões ao dataframe original
        df_original = pd.read_csv(caminho_teste)
        df_original["emissao_prevista"] = previsoes
        
        # Salvar resultado
        nome_arquivo = os.path.basename(caminho_teste)
        caminho_saida = f"./data/output/{nome_arquivo.split('.')[0]}_resultado.csv"
        df_original.to_csv(caminho_saida, index=False)
        
        print(f"✅ Previsões adicionadas e salvas em: {caminho_saida}")
        
    except Exception as e:
        print(f"❌ Erro ao processar arquivo de teste: {e}")

# Verificar argumentos da linha de comando
if __name__ == "__main__":
    # Carregar dados e filtrar N2O
    data_loader = DataLoader("./data/input/br_seeg_emissoes_brasil.csv")
    df_n2o = data_loader.load_data()

    os.makedirs("./data/output", exist_ok=True)

    preprocessor = DataPreprocessor(df_n2o)
    df_n2o = preprocessor.preprocess()

    # Análise exploratória
    exploratory_analysis = ExploratoryAnalysis(df_n2o)
    exploratory_analysis.analyze()

    # Gráficos de Frequência de Emissão por Níveis
    categorias = ["nivel_1", "nivel_2", "nivel_3", "nivel_4", "nivel_5", "nivel_6"]

    for categoria in categorias:
        if categoria in df_n2o.columns:
            niveis = df_n2o[categoria].unique()
            for nivel in niveis:
                plt.figure(figsize=(10, 5))
                subset = df_n2o[df_n2o[categoria] == nivel]
                sns.countplot(data=subset, x="emissao", order=subset["emissao"].value_counts().index)
                plt.xticks(rotation=45)
                plt.title(f"Frequência de Emissão - {categoria}: {nivel}")
                plt.xlabel("Emissão (toneladas)")
                plt.ylabel("Frequência")
                plt.tight_layout()
                plt.savefig(f"./data/output/frequencia_emissao_{categoria}_{nivel}.png")
                plt.close()

    X = df_n2o.drop(columns=["emissao"])
    y = df_n2o["emissao"]

    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    trainer = ModelTrainer(X_train, y_train)
    best_model = trainer.train_xgboost()
    modelo_baseline = trainer.train_baseline()

    evaluator = ModelEvaluator(X_test, y_test)
    mse_xgb = evaluator.evaluate(best_model, "XGBoost")
    mse_baseline = evaluator.evaluate(modelo_baseline, "Regressão Linear (Linha de Base)")
    evaluator.compare_models(mse_xgb, mse_baseline)

    #Gerar CSV de resultado
    df_resultado = pd.DataFrame(X_test, columns=X.columns)
    df_resultado["emissao_real"] = y_test.values
    df_resultado["emissao_prevista"] = best_model.predict(X_test).round(2)

    colunas = df_resultado.columns.tolist()
    colunas_ordenadas = [c for c in ["ano"] if c in colunas] + \
                        [c for c in colunas if c not in ["ano", "emissao_real", "emissao_prevista"]] + \
                        ["emissao_real", "emissao_prevista"]
    df_resultado = df_resultado[colunas_ordenadas]

    df_resultado.to_csv("./data/output/resultado_previsto.csv", index=False)
    print("📁 Arquivo 'resultado_previsto.csv' salvo com sucesso!")

    #Gráfico real vs previsto
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, best_model.predict(X_test), alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Valor real")
    plt.ylabel("Valor previsto")
    plt.title("Comparação: Real vs Previsto (XGBoost)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./data/output/real_vs_previsto.png")
    plt.close()

    # Feature Importance (Gráfico)
    plt.figure(figsize=(12, 6))
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    plt.barh(np.array(X.columns)[indices], importances[indices])
    plt.title("Importância das Variáveis - XGBoost")
    plt.xlabel("Score")
    plt.tight_layout()
    plt.savefig("./data/output/importancia_variaveis.png")
    plt.close()
    
    # Salvar modelo e escalar para uso posterior
    joblib.dump(best_model, "./data/output/modelo_xgboost.pkl")
    joblib.dump(scaler, "./data/output/escalar.pkl")
    print("💾 Modelo e escalar salvos com sucesso!")

    # Verificar se existe arquivo de teste como argumento da linha de comando
    if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
        caminho_teste = sys.argv[1]
        print(f"\n🧪 Processando arquivo de teste: {caminho_teste}")
        processar_arquivo_teste(caminho_teste, best_model, preprocessor, scaler)
