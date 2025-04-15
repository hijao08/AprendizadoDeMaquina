"""
Configuração para testes do pytest.
"""

import sys
import os
from pathlib import Path

# Adiciona o diretório raiz do projeto ao PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent)) 