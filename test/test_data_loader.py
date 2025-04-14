import pandas as pd
from io import StringIO
from src.data_loader import DataLoader
import pytest
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_csv_data():
    csv_data = """gas,emissao,other_column
N2O (t),100,abc
N2O (t),200,def
CO2 (t),300,ghi
N2O (t),,jkl"""
    return pd.read_csv(StringIO(csv_data))

def test_load_data(mock_csv_data):
    # Usar diretamente o mock_csv_data como se fosse o resultado de read_csv
    with patch('pandas.read_csv', return_value=mock_csv_data):
        data_loader = DataLoader("caminho/ficticio.csv")
        result = data_loader.load_data()
        
        # Verificar que o resultado tem as colunas corretas
        assert "gas" not in result.columns
        assert "emissao" in result.columns
        assert "other_column" in result.columns
        
        # Verificar que apenas as linhas com N2O e valores válidos de emissão foram mantidas
        assert len(result) == 2  # Apenas as primeiras duas linhas com N2O e emissão válida
        assert 100.0 in result["emissao"].values
        assert 200.0 in result["emissao"].values