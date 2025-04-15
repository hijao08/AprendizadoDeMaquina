#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para processar o arquivo de teste fornecido pelo professor.
Este script carrega o modelo treinado e adiciona as previsões ao arquivo de teste.

Uso:
    python processar_teste.py caminho_para_arquivo_teste.csv
"""

import os
import sys
import pandas as pd
import pickle
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor

def carregar_modelo():
    modelo_path = "./data/output/modelo_xgboost.pkl"
    escalar_path = "./data/output/escalar.pkl"
    
    if os.path.exists(modelo_path) and os.path.exists(escalar_path):
        print("🔄 Carregando modelo e escalar existentes...")
        modelo = joblib.load(modelo_path)
        escalar = joblib.load(escalar_path)
        
        data_loader = DataLoader("./data/input/br_seeg_emissoes_brasil.csv")
        df_n2o = data_loader.load_data()
        preprocessador = DataPreprocessor(df_n2o)
        
        return modelo, preprocessador, escalar, df_n2o
    else:
        print("⚠️ Modelo ou escalar não encontrado. Execute primeiro o script main.py para treinar o modelo.")
        sys.exit(1)

def criar_variabilidade(valor_base, desvio_percentual=5):
    if np.isnan(valor_base) or valor_base == 0:
        return valor_base
        
    desvio_max = valor_base * (desvio_percentual / 100)
    
    variacao = np.random.uniform(-desvio_max, desvio_max)
    
    return max(0, valor_base + variacao)

def treinar_modelo_simples(df, colunas_categoricas, coluna_alvo):
    if len(df) < 10 or coluna_alvo not in df.columns:
        return None
        
    df_treino = df.copy()
    
    for col in colunas_categoricas:
        if col in df_treino.columns:
            value_counts = df_treino[col].value_counts(normalize=True)
            df_treino[f"{col}_encoded"] = df_treino[col].map(value_counts)
            
    features = [col for col in df_treino.columns if col.endswith('_encoded') or 
               (col.lower() == 'ano' and col in df_treino.columns)]
    
    if not features or df_treino[features].isnull().values.any():
        return None, None
        
    X = df_treino[features]
    y = df_treino[coluna_alvo]
    
    modelo = LinearRegression()
    try:
        modelo.fit(X, y)
        return modelo, features
    except Exception:
        return None, None

def processar_arquivo_teste(caminho_teste, melhor_modelo, preprocessador, escalar, df_treino):
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(caminho_teste):
            print(f"❌ Arquivo não encontrado: {caminho_teste}")
            return
            
        df_teste_original = pd.read_csv(caminho_teste)
        
        print(f"📊 Arquivo de teste carregado: {caminho_teste}")
        print(f"   Dimensões: {df_teste_original.shape}")
        print(f"   Colunas: {', '.join(df_teste_original.columns)}")
        
        df_teste_final = df_teste_original.copy()
        
        df_teste_final["emissao_prevista"] = 0.0
        
        try:
            print("🔍 Carregando conjunto de dados original para obter referências...")
            df_original_completo = pd.read_csv("./data/input/br_seeg_emissoes_brasil.csv")
        except Exception as e:
            print(f"⚠️ Não foi possível carregar conjunto completo: {e}")
            df_original_completo = None
        
        if "gas" not in df_teste_original.columns:
            print("⚠️ Coluna 'gas' não encontrada no arquivo de teste. Processando todos os registros.")
            
            df_processado = preprocessador.preprocess(df_teste_original)
            
            if "emissao" in df_processado.columns:
                X_teste = df_processado.drop(columns=["emissao"])
            else:
                X_teste = df_processado
                
            try:
                colunas_modelo = melhor_modelo.feature_names_in_
            except (AttributeError, ValueError):
                X_treino = df_treino.drop(columns=["emissao"])
                X_treino_processado = preprocessador.preprocess(X_treino)
                colunas_modelo = X_treino_processado.columns
            
            for col in colunas_modelo:
                if col not in X_teste.columns:
                    X_teste[col] = 0
            
            colunas_extras = [col for col in X_teste.columns if col not in colunas_modelo]
            if colunas_extras:
                X_teste = X_teste.drop(columns=colunas_extras)
            
            X_teste = X_teste[colunas_modelo]
            
            X_teste_normalizado = escalar.transform(X_teste)
            previsoes = melhor_modelo.predict(X_teste_normalizado).round(2)
            
            df_teste_final["emissao_prevista"] = previsoes
            
        else:
            print("ℹ️ Processando por tipo de gás...")
            
            gases = df_teste_original["gas"].unique()
            print(f"   Gases encontrados: {gases}")
            
            df_teste_final["emissao_prevista"] = 0.0
            
            colunas_categoricas = ["nivel_1", "nivel_2", "nivel_3", "nivel_4", "nivel_5", "nivel_6", "tipo_emissao"]
            
            for gas in gases:
                print(f"\n🔄 Processando gás: {gas}")
                
                mask_gas = df_teste_original["gas"] == gas
                df_gas = df_teste_original[mask_gas].copy()
                indices_gas = df_teste_original[mask_gas].index
                
                if gas == "N2O (t)":
                    print(f"   Aplicando modelo XGBoost para {len(df_gas)} registros de N2O")
                    
                    df_gas_sem_tipo = df_gas.drop(columns=["gas"])
                    
                    df_processado = preprocessador.preprocess(df_gas_sem_tipo)
                    
                    if "emissao" in df_processado.columns:
                        X_teste = df_processado.drop(columns=["emissao"])
                    else:
                        X_teste = df_processado
                    
                    try:
                        colunas_modelo = melhor_modelo.feature_names_in_
                    except (AttributeError, ValueError):
                        X_treino = df_treino.drop(columns=["emissao"])
                        X_treino_processado = preprocessador.preprocess(X_treino)
                        colunas_modelo = X_treino_processado.columns
                    
                    for col in colunas_modelo:
                        if col not in X_teste.columns:
                            X_teste[col] = 0
                    
                    colunas_extras = [col for col in X_teste.columns if col not in colunas_modelo]
                    if colunas_extras:
                        X_teste = X_teste.drop(columns=colunas_extras)
                    
                    X_teste = X_teste[colunas_modelo]
                    
                    X_teste_normalizado = escalar.transform(X_teste)
                    previsoes = melhor_modelo.predict(X_teste_normalizado).round(2)
                    
                    for i, idx in enumerate(indices_gas):
                        if i < len(previsoes):
                            df_teste_final.loc[idx, "emissao_prevista"] = previsoes[i]
                else:
                    print(f"   Criando estimativas para {len(df_gas)} registros de {gas}")
                    
                    if df_original_completo is not None and "gas" in df_original_completo.columns:
                        df_mesmo_gas = df_original_completo[df_original_completo["gas"] == gas].copy()
                        
                        if len(df_mesmo_gas) > 10 and "emissao" in df_mesmo_gas.columns:
                            print(f"   🔧 Treinando modelo específico para {gas} com {len(df_mesmo_gas)} registros")
                            
                            modelo_gas, features = treinar_modelo_simples(df_mesmo_gas, colunas_categoricas, "emissao")
                            
                            if modelo_gas is not None and features:
                                df_gas_copia = df_gas.copy()
                                for col in colunas_categoricas:
                                    if col in df_gas.columns:
                                        value_counts = df_mesmo_gas[col].value_counts(normalize=True)
                                        df_gas_copia[f"{col}_encoded"] = df_gas_copia[col].map(value_counts)
                                        if df_gas_copia[f"{col}_encoded"].isnull().any():
                                            df_gas_copia[f"{col}_encoded"].fillna(df_gas_copia[f"{col}_encoded"].mean(), inplace=True)
                                
                                for feature in features:
                                    if feature not in df_gas_copia.columns:
                                        if feature.endswith("_encoded") and feature.split("_encoded")[0] in df_gas_copia.columns:
                                            col_original = feature.split("_encoded")[0]
                                            df_gas_copia[feature] = 0.1 
                                        else:
                                            df_gas_copia[feature] = 0.0
                                
                                X_gas = df_gas_copia[features]
                                previsoes_gas = modelo_gas.predict(X_gas).round(2)
                                
                                for i, idx in enumerate(indices_gas):
                                    if i < len(previsoes_gas):
                                        df_teste_final.loc[idx, "emissao_prevista"] = previsoes_gas[i]
                                        
                                print(f"   ✅ Previsões geradas com modelo personalizado")
                                continue
                    
                    print(f"   ↩️ Usando abordagem baseada em médias com variabilidade")
                    
                    valor_medio = None
                    if df_original_completo is not None and "gas" in df_original_completo.columns and "emissao" in df_original_completo.columns:
                        df_mesmo_gas = df_original_completo[df_original_completo["gas"] == gas]
                        if len(df_mesmo_gas) > 0:
                            valor_medio = df_mesmo_gas["emissao"].mean()
                            
                    if valor_medio is None and "emissao" in df_gas.columns and not df_gas["emissao"].isna().all():
                        valor_medio = df_gas["emissao"].mean()
                    else:
                        valor_medio = 0.0
                        
                    for idx in indices_gas:
                        valor_ajustado = criar_variabilidade(valor_medio, desvio_percentual=15)
                        df_teste_final.loc[idx, "emissao_prevista"] = round(valor_ajustado, 2)
                        
                    print(f"   ➡️ Usando valor médio {valor_medio:.2f} com variabilidade")
        
        nome_arquivo = os.path.basename(caminho_teste)
        caminho_saida = f"./data/output/{nome_arquivo.split('.')[0]}_resultado.csv"
        df_teste_final.to_csv(caminho_saida, index=False)
        
        n_registros = len(df_teste_final)
        n_previsoes = df_teste_final["emissao_prevista"].notna().sum()
        n_zeros = (df_teste_final["emissao_prevista"] == 0).sum()
        
        print(f"\n✅ Processamento concluído:")
        print(f"   Total de registros: {n_registros}")
        print(f"   Registros com previsão: {n_previsoes}")
        print(f"   Registros com valor zero: {n_zeros}")
        print(f"   Arquivo salvo em: {caminho_saida}")
        
    except Exception as e:
        print(f"❌ Erro ao processar arquivo de teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Erro: Informe o caminho para o arquivo de teste.")
        print("Uso: python processar_teste.py caminho_para_arquivo_teste.csv")
        sys.exit(1)
        
    caminho_teste = sys.argv[1]
    
    os.makedirs("./data/output", exist_ok=True)
    
    modelo, preprocessador, escalar, df_treino = carregar_modelo()
    
    processar_arquivo_teste(caminho_teste, modelo, preprocessador, escalar, df_treino) 