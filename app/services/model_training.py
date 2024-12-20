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


SETUP = SetupConfig()


def loading_fundamentals(path):
    try:
        logging.info("Carregando os fundamentos.")
        file_path = f"{path}/fundamentals.pkl"
        with open(file_path, "rb") as file:
            loaded_fundamentals = pickle.load(file)
        return loaded_fundamentals
    except Exception as e:
        logging.error(f"Erro no carregamento do fundamentos: {e}")


def create_labels(fundamentals):
    """Criando rótulos: Buy, Hold, Sell

    Não queremos saber quando vender, mas inclui essa categoria para conseguir identificar quando que o nosso modelo vai sugerir uma compra quando na verdade o melhor momento era vender. Isso significa que o modelo errou "mais" do que quando sugeriu comprar e simplesmente o certo era não comprar

    Regra:

    - Subiu mais do que o Ibovespa (ou caiu menos) -> Comprar (Valor = 2)
    - Subiu menos do que o Ibovespa até Ibovespa - 2% (ou caiu mais do que Ibovespa até Ibovespa -2%) -> Não Comprar (Valor = 1)
    - Subiu menos do que o Ibovespa - 2% (ou caiu mais do que Ibovespa -2%) -> Vender (Valor = 0)
    """
    try:
        logging.info("Criando os labels.")
        for company in fundamentals:
            grounds = fundamentals[company]
            grounds = grounds.sort_index()
            for col in grounds:
                if "Adj Close" in col or "IBOV" in col:
                    pass
                else:
                    conditions = [
                        (grounds[col].shift(1) > 0) & (grounds[col] < 0),
                        (grounds[col].shift(1) < 0) & (grounds[col] > 0),
                        (grounds[col].shift(1) < 0) & (grounds[col] < 0),
                        (grounds[col].shift(1) == 0) & (grounds[col] > 0),
                        (grounds[col].shift(1) == 0) & (grounds[col] < 0),
                        (grounds[col].shift(1) < 0) & (grounds[col] == 0),
                    ]
                    conditions_values = [
                        -1,
                        1,
                        (abs(grounds[col].shift(1)) - abs(grounds[col]))
                        / grounds[col].shift(1),
                        1,
                        -1,
                        1,
                    ]
                    grounds[col] = np.select(
                        conditions,
                        conditions_values,
                        default=grounds[col] / grounds[col].shift(1) - 1,
                    )

            grounds["Adj Close"] = (
                grounds["Adj Close"].shift(-1) / grounds["Adj Close"] - 1
            )
            grounds["IBOV"] = grounds["IBOV"].shift(-1) / grounds["IBOV"] - 1
            grounds["Result"] = grounds["Adj Close"] - grounds["IBOV"]
            conditions = [
                (grounds["Result"] > 0),
                (grounds["Result"] <= 0) & (grounds["Result"] >= -0.02),
                (grounds["Result"] < -0.02),
            ]
            conditions_values = [2, 1, 0]
            grounds["Decision"] = np.select(conditions, conditions_values)

            fundamentals[company] = grounds
        return fundamentals
    except Exception as e:
        logging.error(f"Erro ao criar as labels: {e}")


def calculate_empty_values(fundamentals):
    try:
        logging.info("Calculando valores vazios em cada coluna.")
        cols = list(fundamentals["ABEV3"].columns)
        empty_values = dict.fromkeys(cols, 0)
        total_row = 0
        for company in fundamentals:
            table = fundamentals[company]
            total_row += table.shape[0]
            for col in cols:
                quantity_empty_values = table[col].isnull().sum()
                empty_values[col] += quantity_empty_values
        return empty_values, total_row
    except Exception as e:
        logging.error(f"Erro ao calcular valores vazios: {e}")


def process_fundamentals(fundamentals, empty_values, total_row):
    try:
        logging.info("Removendo conlunas com valores vazios.")
        remove_cols = [
            col for col in empty_values if empty_values[col] > (total_row / 3)
        ]
        remove_cols.extend(["Adj Close", "IBOV", "Result"])

        for company in fundamentals:
            fundamentals[company] = fundamentals[company].drop(
                remove_cols, axis=1, errors="ignore"
            )
            fundamentals[company] = fundamentals[company].fillna(0)

        return fundamentals
    except Exception as e:
        logging.error(f"Erro no salvamento do fundamentos: {e}")


def save_fundamentals(fundamentals, path):
    try:
        logging.info("Salvando os fundamentos.")
        file_path = f"{path}/fundamentals_final.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(fundamentals, f)
    except Exception as e:
        logging.error(f"Erro no salvamento do fundamentos: {e}")


def database_fundamentals(fundamentals):
    try:
        logging.info("Salvando os fundamentos.")
        data_frames = []

        for company in fundamentals:
            temp_df = fundamentals[company][1:-1].reset_index(drop=True)
            data_frames.append(temp_df)

        data_base = pd.concat(data_frames, ignore_index=True)
        return data_base
    except Exception as e:
        logging.error(f"Erro no salvamento do fundamentos: {e}")


def saving_data_base(data, file_path):
    try:
        logging.info("Salvando os fundamentos.")
        data.to_parquet(file_path)
    except Exception as e:
        logging.error(f"Erro no salvamento do fundamentos: {e}")


def generate_histogram(data_base, name_image):
    try:
        logging.info("Gerando histograma de Buy, Hold e Sell.")
        # Calcular e logar as contagens normalizadas
        normalized_counts = (
            data_base["Decision"].value_counts(normalize=True).map("{:.1%}".format)
        )
        logging.info(
            f"Proporção de cada valor na coluna 'Decision':\n{normalized_counts}"
        )

        # Criar o histograma
        fig = px.histogram(data_base, x="Decision", color="Decision")

        # Salvar a imagem gerada
        fig.write_image(name_image)

        logging.info(f"Histograma salvo como {name_image}")
    except Exception as e:
        logging.error(f"Erro ao gerar histograma: {e}")


def save_correlation_heatmap(correlations, name_image):
    try:
        logging.info("Gerando e salvando o heatmap de correlações.")
        fig, ax = plt.subplots(figsize=(15, 10))
        sns.heatmap(correlations, cmap="crest", ax=ax)

        fig.savefig(name_image)

        logging.info(f"Heatmap salvo como {name_image}")
    except Exception as e:
        logging.error(f"Erro ao salvar o heatmap: {e}")


def save_high_correlations(correlations, file_name):
    try:
        logging.info("Salvando as correlações.")
        correlations_found = []
        with open(file_name, "w") as file:
            for col in correlations:
                for row in correlations.index:
                    if row != col:
                        value_cor = abs(correlations.loc[row, col])
                        if (
                            value_cor > 0.8
                            and (col, row, value_cor) not in correlations_found
                        ):
                            correlations_found.append((row, col, value_cor))
                            file.write(
                                f"Correlation found: {row} and {col}. Value: {value_cor}\n"
                            )
                            logging.info(
                                f"Correlation found: {row} and {col}. Value: {value_cor}"
                            )
        logging.info(f"Correlações salvas no arquivo {file_name}")
    except Exception as e:
        logging.error(f"Erro ao salvar as correlações: {e}")


def remove_correlations(data_base, correlations):
    try:
        logging.info(f"Removendo correlações")
        data_base = data_base.drop(correlations, axis=1)
        return data_base
    except Exception as e:
        logging.error(f"Erro ao remover as correlações: {e}")


def select_feature(data_base, n_feature):
    try:
        model_tree = ExtraTreesClassifier(random_state=1)
        X = data_base.drop("Decision", axis=1)
        y = data_base["Decision"]
        model_tree.fit(X, y)
        important_features = pd.DataFrame(
            model_tree.feature_importances_, X.columns
        ).sort_values(by=0, ascending=False)
        list_feature = list(important_features.index)[:n_feature]
        logging.info(f"Features selecionadas: {list_feature}")
        return list_feature
    except Exception as e:
        logging.error(f"Erro {e}")


def scale_adjustment(df):
    scaler = StandardScaler()
    df_aux = df.drop("Decision", axis=1)

    df_aux = pd.DataFrame(scaler.fit_transform(df_aux), df_aux.index, df_aux.columns)
    df_aux["Decision"] = df["Decision"]

    return df_aux


def create_feature_dataframe(data_base, list_feature):
    try:
        logging.info("Criando novo DataFrame com as features selecionadas.")
        new_df = scale_adjustment(data_base)
        list_feature.append("Decision")
        new_df = new_df[list_feature].reset_index(drop=True)
        return new_df
    except Exception as e:
        logging.error(f"Erro ao criar novo DataFrame: {e}")


def split_data(df):
    try:
        logging.info("Dividindo os dados em conjuntos de treino e teste.")
        X = df.drop("Decision", axis=1)
        y = df["Decision"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.exception(f"Erro ao dividir os dados: {e}")


def evaluate_model_performance(y_test, prevision, name_model, image_file, report_file):
    try:
        # Exibir o nome do modelo
        logging.info(f"Avaliação do modelo: {name_model}")

        # Gerar e exibir o relatório de classificação
        report = classification_report(y_test, prevision)
        logging.info("Relatório de Classificação:\n", report)
        with open(report_file, "w") as file:
            file.write(f"Avaliação do modelo: {name_model}\n\n")
            file.write("Relatório de Classificação:\n")
            file.write(report)
        logging.info(f"Relatório salvo como {report_file}")

        # Gerar a matriz de confusão
        cf_matrix = pd.DataFrame(
            confusion_matrix(y_test, prevision),
            index=["Sell", "Buy"],
            columns=["Sell", "Buy"],
        )

        # Plotar a matriz de confusão como um heatmap
        plt.figure(figsize=(10, 7))
        sns.heatmap(cf_matrix, annot=True, cmap=sns.color_palette("rocket"), fmt="d")

        # Adicionar títulos ao gráfico
        plt.title(f"Matriz de Confusão - {name_model}")
        plt.xlabel("Previsão")
        plt.ylabel("Verdadeiro")

        # Salvar o gráfico
        plt.savefig(image_file)
        logging.info(f"Gráfico salvo como {image_file}")

    except Exception as e:
        print(f"Erro ao avaliar o modelo: {e}")


def save_model(model, model_name):
    try:
        file_name = f"{SETUP.MODELS_PATH}/{model_name}_model.pkl"
        joblib.dump(model, file_name)
        logging.info(f"Modelo {model_name} salvo como {file_name}")
    except Exception as e:
        logging.error(f"Erro ao salvar o modelo {model_name}: {e}")


def main():
    try:
        logging.info("Iniciando a criação dos modelos.")
        fundamentals = loading_fundamentals(SETUP.PROCESSED_PATH)
        fundamentals = create_labels(fundamentals)
        empty_values, total_row = calculate_empty_values(fundamentals)
        fundamentals = process_fundamentals(fundamentals, empty_values, total_row)
        save_fundamentals(fundamentals, SETUP.PROCESSED_PATH)
        data_base = database_fundamentals(fundamentals)
        data_base_path = f"{SETUP.PROCESSED_PATH}\data_base.parquet"
        saving_data_base(data_base, data_base_path)
        name_image = f"{SETUP.PROCESSED_PATH}/histogram_initial.png"
        generate_histogram(data_base, name_image)
        data_base.loc[data_base["Decision"] == 1, "Decision"] = 0
        name_image = f"{SETUP.PROCESSED_PATH}/histogram_final.png"
        generate_histogram(data_base, name_image)
        correlations = data_base.corr()
        name_image = f"{SETUP.PROCESSED_PATH}/correlations_initial.png"
        save_correlation_heatmap(correlations, name_image)
        file_name = f"{SETUP.PROCESSED_PATH}/correlations_initial.txt"
        save_high_correlations(correlations, file_name)
        data_base = remove_correlations(data_base, SETUP.REMOVE_CORRELATIONS)
        name_image = f"{SETUP.PROCESSED_PATH}/correlations_final.png"
        save_correlation_heatmap(correlations, name_image)
        file_name = f"{SETUP.PROCESSED_PATH}/correlations_final.txt"
        save_high_correlations(correlations, file_name)
        list_features = select_feature(data_base, SETUP.N_FEATURES)
        df = create_feature_dataframe(data_base, list_features)
        df_path = f"{SETUP.PROCESSED_PATH}\df_final.parquet"
        saving_data_base(df, df_path)
        X_train, X_test, y_train, y_test = split_data(df)
        logging.info("Treinando o Dummy para ser nossa baseline.")
        dummy = DummyClassifier(strategy="stratified", random_state=1)
        dummy.fit(X_train, y_train)
        prevision_dummy = dummy.predict(X_test)
        image_file = f"{SETUP.PROCESSED_PATH}/dummy.png"
        report_file = f"{SETUP.PROCESSED_PATH}/dummy.txt"
        evaluate_model_performance(
            y_test, prevision_dummy, "Dummy", image_file, report_file
        )
        models = SETUP.MODELS
        logging.info("Treinando diversos modelos.")
        for model_name in models:
            model = models[model_name]
            model.fit(X_train, y_train)
            prevision = model.predict(X_test)
            image_file = f"{SETUP.PROCESSED_PATH}/{model_name}.png"
            report_file = f"{SETUP.PROCESSED_PATH}/{model_name}.txt"
            evaluate_model_performance(
                y_test, prevision, model_name, image_file, report_file
            )
            models[model_name] = model
            save_model(model, model_name)

    except Exception as e:
        logging.error(f"Erro no salvamento do fundamentos: {e}")
