import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pandas as pd
import logging
from app.utils.logger import configure_logging
from setup import SetupConfig


def is_valid_file(file_path, column_threshold=40):
    """
    Verifica se um arquivo Excel é válido com base no número de colunas.

    Args:
        file_path (str): Caminho do arquivo a ser verificado.
        column_threshold (int): Número mínimo de colunas para o arquivo ser válido.

    Returns:
        bool: True se o arquivo for válido, False caso contrário.
    """
    df = pd.read_excel(file_path)
    return df.shape[1] > column_threshold


def check_balance(directory_path, column_threshold=40, delete_invalid=False):
    """
    Analisa arquivos Excel no diretório especificado, verificando sua validade.

    Args:
        directory_path (str): Caminho do diretório contendo os arquivos.
        column_threshold (int): Número mínimo de colunas para um arquivo ser considerado válido.
        delete_invalid (bool): Se True, arquivos inválidos serão excluídos.

    Returns:
        tuple: Lista de arquivos válidos e inválidos.
    """
    valid_files = []
    invalid_files = []

    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)

        if file_name.endswith(".xls") and os.path.isfile(file_path):
            if is_valid_file(file_path, column_threshold):
                valid_files.append(file_name)
            else:
                invalid_files.append(file_name)
                if delete_invalid:
                    os.remove(file_path)

    return valid_files, invalid_files


if __name__ == "__main__":
    # Instância da configuração
    SETUP = SetupConfig()
    print(SETUP.DOWNLOAD_PATH)
    # # Configuração do diretório de trabalho
    # current_directory = str(fr'{os.getcwd()}\data\raw')
    # print(current_directory)
    # # Executa a verificação dos arquivos
    valid_files, invalid_files = check_balance(
        directory_path=SETUP.RAW_PATH,
        column_threshold=1,
        delete_invalid=False,  # Define se arquivos inválidos serão removidos
    )

    # Exibe os resultados
    print(f"Arquivos válidos ({len(valid_files)}): {valid_files}")
    print(f"Arquivos inválidos ({len(invalid_files)}): {invalid_files}")
