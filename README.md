# Aprendizado de Máquina: Análise de Emissões de N2O no Brasil

Este projeto realiza uma análise das emissões de N2O no Brasil utilizando técnicas de aprendizado de máquina. O objetivo é prever as emissões futuras com base em dados históricos.

## **Requisitos**

**Este projeto requer o Python 3.12. Certifique-se de que esta versão está instalada em seu sistema antes de prosseguir com a configuração do ambiente.**

## Estrutura do Projeto

- `src/main.py`: Script principal que executa o fluxo de análise e previsão.
- `src/processar_teste.py`: Script para adicionar previsões a um arquivo de teste.
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
python src/main.py
```

Para adicionar previsões a um arquivo de teste:

```bash
# Primeiro, execute o script principal para treinar o modelo
python src/main.py

# Depois, processe o arquivo de teste
python src/processar_teste.py br_seeg_emissoes_brasil.csv
```

Após a execução, um novo arquivo será gerado na pasta `data/output` com o nome do arquivo original seguido de `_resultado.csv`, contendo as previsões do modelo.

### Processamento de Arquivos de Teste

O script `processar_teste.py` foi projetado para processar arquivos de teste fornecidos, adicionando previsões do modelo treinado. Ele possui as seguintes funcionalidades:

**Previsão para N2O**: Para registros de N2O, utiliza o modelo XGBoost treinado com alta precisão.

## Testes

Para executar os testes automatizados do projeto, utilize o pytest:

```bash
# Executar todos os testes
python -m pytest test/

# Executar testes com detalhes
python -m pytest test/ -v

# Executar um arquivo de teste específico
python -m pytest test/test_data_loader.py

# Executar um teste específico
python -m pytest test/test_data_loader.py::test_load_data
```

Certifique-se de que o pytest está instalado no ambiente virtual:

```bash
pip install pytest
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
