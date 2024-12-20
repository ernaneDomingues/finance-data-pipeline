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


def split_data(df):
    try:
        logging.info("Dividindo os dados em conjuntos de treino e teste.")
        X = df.drop("Decision", axis=1)
        y = df["Decision"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.exception(f"Erro ao dividir os dados: {e}")


def save_model(model, model_name):
    try:
        file_name = f"{SETUP.MODELS_PATH}/{model_name}_model.pkl"
        joblib.dump(model, file_name)
        logging.info(f"Modelo {model_name} salvo como {file_name}")
    except Exception as e:
        logging.error(f"Erro ao salvar o modelo {model_name}: {e}")


# Configurar scorer personalizado
precision_personal = make_scorer(
    precision_score,
    pos_label=2,
    average="binary",
    zero_division=0,  # Lidar com ausência de rótulos
)

# Configurar validação cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Definir grid de parâmetros (padrão genérico para todos os modelos)
param_grids = {
    "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]},
    "Decision_Tree": {"criterion": ["gini", "entropy"], "max_depth": [None, 5, 10]},
    "Random_Forest": {
        "n_estimators": [50, 100],
        "max_depth": [None, 10],
        "max_features": ["auto", "sqrt"],
    },
    "ExtraTree": {"n_estimators": [50, 100], "max_depth": [None, 10]},
    "Gradient_Boost": {
        "learning_rate": [0.01, 0.1],
        "n_estimators": [50, 100],
    },
    "KNN": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
    "Logistic_Regression": {"C": [0.01, 0.1, 1, 10]},
    "Naive_Bayes": {},  # Sem parâmetros para ajustar
    "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
    "Rede_Neural": {"hidden_layer_sizes": [(50,), (100,)], "alpha": [0.0001, 0.001]},
}


# Função para realizar GridSearchCV
def run_grid_search(model_name, model, param_grid, X_train, y_train):
    print(f"\nExecutando GridSearch para {model_name}...\n")
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=precision_personal,
        cv=cv,
    )
    grid_result = grid.fit(X_train, y_train)
    print(f"Melhores parâmetros para {model_name}: {grid_result.best_params_}")
    print(f"Melhor pontuação para {model_name}: {grid_result.best_score_:.4f}")
    return grid_result


def main():
    try:
        logging.info("Realizando o GridSearch do modelo.")
        df_path = f"{SETUP.PROCESSED_PATH}\df_final.parquet"
        df = loading_database(df_path)
        X_train, X_test, y_train, y_test = split_data(df)
        model_select = SETUP.MODEL_SELECT
        model_file = f"{SETUP.MODELS_PATH}/{model_select}.pkl"
        model = loading_model(model_file)
        param_grid = param_grids[model_select]
        result = run_grid_search(model_select, model, param_grid, X_train, y_train)
        model_tuning = result.best_estimator_
        model_name = f"{model_select}_tuning"
        save_model(model_tuning, model_name)
    except Exception as e:
        logging.error(f"Erro ao realizar o GridSearch: {e}")
