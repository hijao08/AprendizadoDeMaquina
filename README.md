# Aprendizado de Máquina: Análise de Emissões de N2O no Brasil

Este projeto realiza uma análise das emissões de N2O no Brasil utilizando técnicas de aprendizado de máquina. O objetivo é prever as emissões futuras com base em dados históricos.

## **Requisitos**

**Este projeto requer o Python 3.12. Certifique-se de que esta versão está instalada em seu sistema antes de prosseguir com a configuração do ambiente.**

## Estrutura do Projeto

- `main.py`: Script principal que executa o fluxo de análise e previsão.
- `data/input`: Contém os dados de entrada.
- `data/output`: Armazena os resultados e gráficos gerados.

## Configuração do Ambiente

Para criar o ambiente virtual, abra o terminal dentro da pasta do projeto e execute:

```bash
python3 -m venv ambiente_virtual
```

Ative o ambiente virtual:

```bash
source ambiente_virtual/bin/activate
```

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

Para desativar o ambiente virtual:

```bash
deactivate
```

## Uso

Execute o script principal para realizar a análise e gerar previsões:

```bash
python main.py
```

## Dados

Os dados utilizados neste projeto são provenientes do conjunto de dados de emissões do Brasil, filtrados para incluir apenas o gás N2O.

## Modelos

O projeto utiliza dois modelos principais para previsão:

- **XGBoost**: Um modelo de árvore de decisão otimizado para previsão de emissões.
- **Regressão Linear**: Usado como linha de base para comparação de desempenho.

## Resultados

Os resultados das previsões são salvos em `data/output/resultado_previsto.csv`. Gráficos de análise e importância das variáveis também são gerados nesta pasta.

## Contribuição

Sinta-se à vontade para contribuir com melhorias para este projeto. Faça um fork do repositório e envie suas sugestões através de pull requests.
