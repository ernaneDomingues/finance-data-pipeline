import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from pathlib import Path
import logging
from app.utils.functions import check_balance
from app.utils import logger
from setup import SetupConfig
import pandas as pd
import numpy as np
import pickle


SETUP = SetupConfig()

print(SETUP.RAW_PATH)
print(SETUP.PROCESSED_PATH)

valid_files, invalid_files = check_balance(
    directory_path=SETUP.RAW_PATH,
    column_threshold=4,
    delete_invalid=False,
)


def file_processing(raw_path):
    try:
        logging.info("Processando os fundamentos.")
        fundamentals = {}

        for file in valid_files:
            ticket = file.replace("bal_", "").replace(".xls", "")
            print(ticket)

            balance = pd.read_excel(f"{raw_path}/{file}", sheet_name=0)
            balance.iloc[0, 0] = ticket
            balance.columns = balance.iloc[0]
            balance = balance[1:]

            dre = pd.read_excel(f"{raw_path}/{file}", sheet_name=1)
            dre.iloc[0, 0] = ticket
            dre.columns = dre.iloc[0]
            dre = dre[1:]

            combined = pd.concat([balance.set_index(ticket), dre.set_index(ticket)])
            fundamentals[ticket] = combined

        return fundamentals
    except Exception as e:
        logging.error(f"Erro no processamento dos fundamentos: {e}")


def loading_quotes(raw_path):
    try:
        logging.info("Carregando as cotações.")
        quote_df = pd.read_excel(f"{raw_path}/quotations.xlsx")
        quote_df["Ticker"] = quote_df["Ticker"].str.replace(".SA", "", regex=False)
        quote = {}
        for ticker in quote_df["Ticker"].unique():
            quote[ticker] = quote_df.loc[quote_df["Ticker"] == ticker, :]

        df_ibov = quote_df[quote_df["Ticker"] == "^BVSP"]
        return quote, df_ibov
    except Exception as e:
        logging.error(f"Erro no carregamento das cotações: {e}")


def verify_quote(fundamentals, quote):
    try:
        logging.info("Verificando empresas que não têm cotação.")
        for ticker in valid_files:
            ticker = ticker.replace("bal_", "").replace(".xls", "")
            if quote[ticker].isnull().values.any():
                quote.pop(ticker)
                fundamentals.pop(ticker)
        return fundamentals, quote
    except Exception as e:
        logging.error(f"Erro na verificação: {e}")


def join_fundamentals_quote(fundamentals, quote):
    try:
        logging.info("Juntando as cotações aos fundamentos.")
        for company in fundamentals:
            table = fundamentals[company].T
            table.index = pd.to_datetime(table.index, format="%d/%m/%Y")
            table_quote = quote[company].set_index("Date")
            table_quote = table_quote[["Adj Close"]]
            table = table.merge(table_quote, right_index=True, left_index=True)
            table.index.name = company
            fundamentals[company] = table
        return fundamentals
    except Exception as e:
        logging.error(f"Erro no join das cotações ao fundamentos: {e}")


def filtering_fundamentals(fundamentals):
    try:
        logging.info("Filtrando os fundamentos.")
        cols = list(fundamentals["ABEV3"].columns)
        companies = list(fundamentals.keys())
        print(len(fundamentals))

        for company in companies:
            if set(cols) != set(fundamentals[company].columns):
                fundamentals.pop(company)

        print(len(fundamentals))
        return fundamentals, cols
    except Exception as e:
        logging.error(f"Erro no join das cotações ao fundamentos: {e}")


def verify_cols_name(cols):
    try:
        logging.info(
            "Verificando os nomes das colunas e renomeando as colunas duplicadas."
        )
        text_cols = ";".join(cols)

        modified_cols = []

        for col in cols:
            if cols.count(col) == 2 and col not in modified_cols:
                text_cols = text_cols.replace(f";{col};", f";{col}_1;", 1)
                modified_cols.append(col)

        cols = text_cols.split(";")
        return cols
    except Exception as e:
        logging.error(f"Erro na verificação das colunas: {e}")


def apply_cols_name(fundamentals, cols):
    try:
        logging.info("Aplicando os novos nomes das colunas.")
        for company in fundamentals:
            fundamentals[company].columns = cols
        return fundamentals
    except Exception as e:
        logging.error(f"Erro na renomeação das colunas: {e}")


def verify_empty_value(fundamentals, cols):
    try:
        logging.info("Verificando os valores vazios.")
        empty_values = dict.fromkeys(cols, 0)
        total_row = 0
        for company in fundamentals:
            table = fundamentals[company]
            total_row += table.shape[0]
            for col in cols:
                quantity_empty_values = pd.isnull(table[col].sum())
                empty_values[col] += quantity_empty_values

        return empty_values
    except Exception as e:
        logging.error(f"Erro na verificação de valores vazios: {e}")


def clean_cols(fundamentals, empty_values):
    try:
        logging.info("Limpando as colunas vazias.")
        remove_cols = []
        for col in empty_values:
            if empty_values[col] > 50:
                remove_cols.append(col)

        for company in fundamentals:
            fundamentals[company] = fundamentals[company].drop(remove_cols, axis=1)
            fundamentals[company] = fundamentals[company].ffill()

        return fundamentals
    except Exception as e:
        logging.error(f"Erro na remoção de colunas com valores vazios: {e}")


def adjusted_df_ibov(fundamentals, df_ibov):
    try:
        logging.info("Ajustando o df_ibov.")
        dates = fundamentals["ABEV3"].index
        df_ibov.index = pd.to_datetime(df_ibov.index, errors="coerce")
        dates = pd.to_datetime(dates, errors="coerce")
        for date in dates:
            if date not in df_ibov.index:

                df_ibov.loc[date] = np.nan

        df_ibov = df_ibov.sort_index()
        df_ibov = df_ibov.ffill()
        df_ibov = df_ibov.rename(columns={"Adj Close": "IBOV"})
        return df_ibov
    except Exception as e:
        logging.error(f"Erro no ajustes do df_ibov: {e}")


def join_fundamentals_ibov(fundamentals, df_ibov):
    try:
        logging.info("Juntando ao fundamentos a cotação do IBOV.")
        for company in fundamentals:
            fundamentals[company] = fundamentals[company].merge(
                df_ibov[["IBOV"]], left_index=True, right_index=True
            )

        return fundamentals
    except Exception as e:
        logging.error(f"Erro no join do IBOV ao fundamentos: {e}")


def save_fundamentals(fundamentals, path):
    try:
        logging.info("Salvando os fundamentos.")
        file_path = f"{path}/fundamentals.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(fundamentals, f)
    except Exception as e:
        logging.error(f"Erro no salvamento do fundamentos: {e}")


def main(fundamentals):
    logging.info("Iniciando processamento dos dados...")
    try:
        fundamentals = file_processing(SETUP.RAW_PATH)
        df_quotes, df_ibov = loading_quotes(SETUP.RAW_PATH)
        fundamentals, df_quotes = verify_quote(fundamentals, df_quotes)
        fundamentals = join_fundamentals_quote(fundamentals, df_quotes)
        fundamentals, cols = filtering_fundamentals(fundamentals)
        cols = verify_cols_name(cols)
        fundamentals = apply_cols_name(fundamentals, cols)
        empty_values = verify_empty_value(fundamentals, cols)
        fundamentals = clean_cols(fundamentals, empty_values)
        df_ibov = adjusted_df_ibov(df_ibov)
        fundamentals = join_fundamentals_ibov(fundamentals, df_ibov)
        save_fundamentals(fundamentals, SETUP.PROCESSED_PATH)
    except Exception as e:
        logging.error(f"Erro no processamento dos dados: {e}")
