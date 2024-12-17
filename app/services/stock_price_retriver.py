import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import yfinance as yf
import pandas as pd
from app.utils.functions import check_balance
from datetime import datetime
from openpyxl.workbook import Workbook
from setup import SetupConfig
from pathlib import Path
import logging
from app.utils.logger import configure_logging

SETUP = SetupConfig()

START_DAY = "2000-01-01"
END_DAY = datetime.today().strftime("%Y-%m-%d")


def get_company_tickers(path):
    logging.info("Pegando os tickers das empresas validadas.")
    ticket_default, _ = check_balance(
        directory_path=path, column_threshold=1, delete_invalid=False
    )
    company_tickets = [
        ticket.replace("bal_", "").replace(".xls", ".SA") for ticket in ticket_default
    ]
    return company_tickets


def stock_price_retriver(tickers, start_date, end_date, output_file):
    """
    Baixa cotações de ações de uma lista de tickers usando yfinance e salva em um arquivo Excel.

    Args:
        tickers (list): Lista de tickers (códigos das empresas).
        start_date (str): Data de início no formato 'YYYY-MM-DD'.
        end_date (str): Data de fim no formato 'YYYY-MM-DD'.
        output_file (str): Nome do arquivo Excel para salvar os dados.
    """
    all_data = []

    for ticker in tickers:
        print(f"Baixando dados de {ticker}...")
        try:
            # Baixa os dados da ação no período especificado
            df = yf.download(ticker, start=start_date, end=end_date)
            
            if not df.empty:
                # Adiciona o nome do ticker em uma coluna para identificação
                df['Ticker'] = ticker
                # Remove MultiIndex completamente
                df = df.reset_index()
                df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]
                all_data.append(df)
            else:
                print(f"Dados indisponíveis para {ticker}.")
        except Exception as e:
            print(f"Erro ao processar {ticker}: {e}")

    if all_data:
        # Concatena todos os DataFrames
        result = pd.concat(all_data, ignore_index=True)
        # Exporta para Excel
        result.to_excel(output_file, index=False, engine="openpyxl")
        print(f"Dados exportados com sucesso para {output_file}.")
    else:
        print("Nenhum dado foi exportado, pois não foram encontrados dados válidos.")


if __name__ == "__main__":
    company_tickers = get_company_tickers(SETUP.RAW_PATH)
    file_output = fr'{SETUP.RAW_PATH}\quotations.xlsx'
    stock_price_retriver(company_tickers, START_DAY, END_DAY, file_output)
