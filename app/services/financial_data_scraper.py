import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import time
import zipfile
from pathlib import Path
import logging
from app.utils.logger import configure_logging
from setup import SetupConfig

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service
from webdriver_manager.microsoft import EdgeChromiumDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SETUP = SetupConfig()

service = Service(EdgeChromiumDriverManager().install())
driver = webdriver.Edge(service=service)

def rename_file(path, new_name_file):
    try:
        file_path = os.path.join(path, "balanco.xls")
        new_file_path = os.path.join(path, new_name_file)
        
        if os.path.exists(file_path):
            os.rename(file_path, new_file_path)
            logging.info(f"Arquivo renomeado para {new_name_file}")
            print(f"Arquivo renomeado para {new_name_file}")
        else:
            logging.info("Arquivo 'balanco.xls' não encontrado.")
            print("Arquivo 'balanco.xls' não encontrado.")
    except Exception as e:
        logging.error(f"Erro ao processar {file}: {e}")
        print(f"Ocorreu um erro ao renomear o arquivo: {e}")


def unzip_file(downloads_path, target_path):
    try:
        files = os.listdir(downloads_path)
        for file in files:
            if file.startswith('bal_') and file.endswith(".zip"):
                file_path = os.path.join(downloads_path, file)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(target_path)
                os.remove(file_path)
                logging.info(f"{file} extraido com sucesso!")
    except FileNotFoundError as e:
        logging.error(f"Erro ao processar {file}: {e}")
        print(f"Erro: Arquivo ou diretório não encontrado - {e}")
    except zipfile.BadZipFile as e:
        logging.error(f"Erro ao processar {file}: {e}")
        print(f"Erro: Arquivo ZIP corrompido - {e}")
    except Exception as e:
        logging.error(f"Erro ao processar {file}: {e}")
        print(f"Ocorreu um erro inesperado: {e}")



def extraction_balance(ticket_list):
    logging.info("Iniciando o processo de web scraping...")
    for ticket in ticket_list:
        logging.info(f"Processando ticket: {ticket}")
        driver.get(
            f"https://www.fundamentus.com.br/balancos.php?papel={ticket}&interface=mobile#"
        )
        time.sleep(3)
        # driver.maximize_window()
        try:
            # Esperar pelo botão de download
            download_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, 'bt-baixar'))
            )
            download_button.click()
            time.sleep(5)  # Ajustar conforme necessário
            unzip_file(downloads_path=SETUP.DOWNLOAD_PATH, target_path=SETUP.RAW_PATH)
            time.sleep(3)
            new_name = f"bal_{ticket}.xls"
            rename_file(path=SETUP.RAW_PATH, new_name_file=new_name)
        except Exception as e:
            logging.error(f"Erro ao processar {ticket}: {e}")
            print(f"Erro ao processar o ticket {ticket}: {e}")
    logging.info("Fim do processo de web scraping...")


if __name__ == "__main__":
    try:
        with open("data/ticker_br.txt", "r") as f:
            ticket_br = [line.strip() for line in f]
        extraction_balance(ticket_br)
    finally:
        driver.quit()  # Garantir que o driver seja encerrado
