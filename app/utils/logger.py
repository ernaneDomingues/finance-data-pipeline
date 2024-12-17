import logging
import os

def configure_logging():
    """
    Configura o sistema de logging para o projeto.
    Salva os logs na pasta `logs` na raiz do projeto e também os exibe no console.
    """
    # Caminho para a pasta de logs
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    
    # Certifica-se de que a pasta de logs existe
    os.makedirs(log_dir, exist_ok=True)
    
    # Define o caminho do arquivo de log
    log_file = os.path.join(log_dir, "app.log")
    
    # Configura o logging
    logging.basicConfig(
        level=logging.INFO,  # Define o nível de log
        format="%(asctime)s - %(levelname)s - %(message)s",  # Formato do log
        handlers=[
            logging.FileHandler(log_file),  # Logs no arquivo
            logging.StreamHandler()        # Logs no console
        ]
    )

# Chama a configuração ao importar o módulo
configure_logging()
