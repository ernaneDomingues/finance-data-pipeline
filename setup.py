from pathlib import Path
import os

class SetupConfig:
    """Classe de configuração para os paths do projeto."""

    def __init__(self):
        self.DOWNLOAD_PATH = str(Path.home() / "Downloads")

    @property
    def RAW_PATH(self):
        """Atributo dinâmico que retorna o caminho 'data/raw' no diretório atual."""
        return str(Path(os.getcwd()) / "data" / "raw")


