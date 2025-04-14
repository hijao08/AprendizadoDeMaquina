import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.exploratory_analysis import ExploratoryAnalysis
from io import StringIO
from src.data_loader import DataLoader

@pytest.fixture
def sample_dataframe():
    """Cria um DataFrame de exemplo para os testes."""
    data = {
        "emissao": [10.5, 20.3, 30.1, 40.2, 50.0],
        "nivel_1": ["A", "B", "A", "C", "B"],
        "nivel_2": ["X", "Y", "X", "Z", "Y"],
    }
    return pd.DataFrame(data)

@patch("src.exploratory_analysis.plt")
@patch("src.exploratory_analysis.sns")
@patch("src.exploratory_analysis.os.makedirs")
def test_analyze(mock_makedirs, mock_sns, mock_plt, sample_dataframe):
    """Testa o método analyze da classe ExploratoryAnalysis."""
    # Mock para evitar criação de arquivos
    mock_makedirs.return_value = None
    mock_plt.savefig = MagicMock()

    # Instancia a classe com o DataFrame de exemplo
    analysis = ExploratoryAnalysis(sample_dataframe)

    # Executa o método analyze
    analysis.analyze()

    # Verifica se os gráficos foram gerados
    assert mock_plt.figure.call_count > 0
    assert mock_plt.savefig.call_count > 0

    # Verifica se os gráficos de seaborn foram chamados
    assert mock_sns.histplot.called
    assert mock_sns.countplot.called

    # Verifica se a matriz de correlação foi gerada
    assert mock_sns.heatmap.called