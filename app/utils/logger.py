import logging
import os
import inspect

def configure_logging():
    """
    Configura o sistema de logging criando um arquivo específico com base no nome do módulo que chamou.
    """
    # Caminho para a pasta de logs
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)  # Garante que a pasta de logs exista

    # Inspeciona a pilha para encontrar o módulo correto (ignorando entradas inválidas)
    caller_filename = None
    for frame in inspect.stack():
        if frame.filename != __file__ and os.path.isfile(frame.filename):
            caller_filename = frame.filename  # Caminho completo do arquivo chamador
            break

    # Garante que um nome válido foi encontrado
    if not caller_filename:
        caller_name = "unknown"
    else:
        caller_name = os.path.splitext(os.path.basename(caller_filename))[0]

    # Define o caminho do arquivo de log específico
    log_file = os.path.join(log_dir, f"{caller_name}.log")

    # Configura o logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Log em arquivo específico
            logging.StreamHandler()         # Log no console
        ]
    )

# Chama a configuração ao importar o módulo
configure_logging()
