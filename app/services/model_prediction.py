import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from pathlib import Path
import logging
from app.utils.functions import check_balance
from app.utils import logger
from model_training import remove_correlations, select_feature, scale_adjustment
from setup import SetupConfig
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, precision_score

SETUP = SetupConfig()


def loading_model(file_path):
    try:
        logging.info("Carregando o modelo.")
        with open(file_path, "rb") as file:
            model_tree = joblib.load(file)
        return model_tree
    except Exception as e:
        logging.error(f"Erro ao carregar o modelo: {e}")


def loading_database(file_path):
    try:
        logging.info("Carregando o DataBbase.")
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        logging.error(f"Erro ao carregar o DataBase")


def loading_fundamentals(path):
    try:
        logging.info("Carregando os fundamentos.")
        file_path = f"{path}/fundamentals_final.pkl"
        with open(file_path, "rb") as file:
            loaded_fundamentals = pickle.load(file)
        return loaded_fundamentals
    except Exception as e:
        logging.error(f"Erro no carregamento do fundamentos: {e}")


def main():
    try:
        model_path = f"{SETUP.MODELS_PATH}\{SETUP.MODEL_SELECT}_tuning_model.pkl"
        model_tuning = loading_model(model_path)
        df_path = f"{SETUP.PROCESSED_PATH}\data_base.parquet"
        df = loading_database(df_path)
        df = remove_correlations(df, SETUP.REMOVE_CORRELATIONS)
        df.loc[df["Decision"] == 1, "Decision"] = 0
        list_features = select_feature(df, SETUP.N_FEATURES)
        ult_tri_fundamentos = loading_fundamentals(SETUP.PROCESSED_PATH)
        data_frames = []
        lista_empresas = []
        for empresa in ult_tri_fundamentos:
            temp_df = ult_tri_fundamentos[empresa][-1:].reset_index(drop=True)
            data_frames.append(temp_df)
            lista_empresas.append(empresa)
        ult_tri_base_dados = pd.concat(data_frames, ignore_index=True)
        ult_tri_base_dados = ult_tri_base_dados.reset_index(drop=True)
        ult_tri_base_dados = ult_tri_base_dados[list_features]
        ult_tri_base_dados = scale_adjustment(ult_tri_base_dados)
        ult_tri_base_dados = ult_tri_base_dados.drop("Decision", axis=1)
        previsoes_ult_tri = model_tuning.predict(ult_tri_base_dados)
        for index, value in enumerate(previsoes_ult_tri):
            if value == 2:
                logging.info(f"{lista_empresas[index]}, Buy")
            elif value == 0:
                logging.info(f"{lista_empresas[index]}, Sell")
    except Exception as e:
        logging.error(f"Erro {e}")
