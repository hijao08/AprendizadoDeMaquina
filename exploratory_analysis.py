import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ExploratoryAnalysis:
    def __init__(self, df):
        self.df = df

    def analyze(self):
        print("\n📌 Dimensões do dataframe filtrado (apenas N2O):")
        print(self.df.shape)
        print("\n🧾 Primeiras linhas:")
        print(self.df.head())
        print("\n🔍 Tipos de dados:")
        print(self.df.dtypes)
        print("\n❓ Valores nulos:")
        print(self.df.isnull().sum())
        print("\n📊 Estatísticas da emissão:")
        print(self.df["emissao"].describe())

        # Gráfico: distribuição da emissão
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df["emissao"], bins=30, kde=True)
        plt.title("Distribuição dos valores de emissão (N2O)")
        plt.xlabel("Emissão (toneladas)")
        plt.ylabel("Frequência")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./data/output/distribuicao_emissao.png")
        plt.close()

        # Frequência das categorias em nivel_1
        if "nivel_1" in self.df.columns:
            plt.figure(figsize=(10, 4))
            sns.countplot(data=self.df, x="nivel_1", order=self.df["nivel_1"].value_counts().index)
            plt.xticks(rotation=45)
            plt.title("Frequência por categoria - nivel_1")
            plt.tight_layout()
            plt.savefig("./data/output/frequencia_nivel1.png")
            plt.close()
            
        # Frequência das categorias em nivel_2
        if "nivel_2" in self.df.columns:
            plt.figure(figsize=(10, 4))
            sns.countplot(data=self.df, x="nivel_2", order=self.df["nivel_2"].value_counts().index)
            plt.xticks(rotation=45)
            plt.title("Frequência por categoria - nivel_2")
            plt.tight_layout()
            plt.savefig("./data/output/frequencia_nivel2.png")
            plt.close()
            
        # Frequência das categorias em nivel_3
        if "nivel_3" in self.df.columns:
            plt.figure(figsize=(10, 4))
            sns.countplot(data=self.df, x="nivel_3", order=self.df["nivel_3"].value_counts().index)
            plt.xticks(rotation=45)
            plt.title("Frequência por categoria - nivel_3") 
            plt.tight_layout()
            plt.savefig("./data/output/frequencia_nivel3.png")
            plt.close()
            
        # Frequência das categorias em nivel_4
        if "nivel_4" in self.df.columns:
            plt.figure(figsize=(10, 4))
            sns.countplot(data=self.df, x="nivel_4", order=self.df["nivel_4"].value_counts().index)
            plt.xticks(rotation=45)
            plt.title("Frequência por categoria - nivel_4") 
            plt.tight_layout()
            plt.savefig("./data/output/frequencia_nivel4.png")
            plt.close()
            
        # Frequência das categorias em nivel_5
        if "nivel_5" in self.df.columns:    
            plt.figure(figsize=(10, 4))
            sns.countplot(data=self.df, x="nivel_5", order=self.df["nivel_5"].value_counts().index)
            plt.xticks(rotation=45)
            plt.title("Frequência por categoria - nivel_5") 
            plt.tight_layout()
            plt.savefig("./data/output/frequencia_nivel5.png")  
            plt.close()
            
        # Frequência das categorias em nivel_6
        if "nivel_6" in self.df.columns:
            plt.figure(figsize=(10, 4))
            sns.countplot(data=self.df, x="nivel_6", order=self.df["nivel_6"].value_counts().index) 
            plt.xticks(rotation=45)
            plt.title("Frequência por categoria - nivel_6") 
            plt.tight_layout()
            plt.savefig("./data/output/frequencia_nivel6.png")
            plt.close()

        # Correlação numérica
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.df.select_dtypes(include=[np.number]).corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Matriz de Correlação (variáveis numéricas)")
        plt.tight_layout()
        plt.savefig("./data/output/correlacao_numerica.png")
        plt.close() 