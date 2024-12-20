from pathlib import Path
import os
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


class SetupConfig:
    """Classe de configuração para os paths do projeto."""

    def __init__(self):
        self.DOWNLOAD_PATH = str(Path.home() / "Downloads")

    @property
    def RAW_PATH(self):
        """Atributo dinâmico que retorna o caminho 'data/raw' no diretório atual."""
        return str(Path(os.getcwd()) / "data" / "raw")

    @property
    def PROCESSED_PATH(self):
        """Atributo dinâmico que retorna o caminho 'data/processed' no diretório atual."""
        return str(Path(os.getcwd()) / "data" / "processed")

    @property
    def MODELS_PATH(self):
        """Atributo dinâmico que retorna o caminho 'models' no diretório atual."""
        return str(Path(os.getcwd()) / "models")

    @property
    def REMOVE_CORRELATIONS(self):
        """Atributo dinâmico que retorna a lista das colunas que tem correlação maior que 0.8."""
        return [
            "Passivo Total",
            "Provisões",
            "Receita Líquida de Vendas e/ou Serviços",
            "Intangível",
        ]

    @property
    def N_FEATURES(self):
        """Atributo dinâmico que retorna a quantidade de feature selection desejado."""
        return 15

    @property
    def MODELS(self):
        """Atributo dinâmico que retorna a lista de modelos"""
        return {
            "AdaBoost": AdaBoostClassifier(random_state=1),
            "Decision_Tree": DecisionTreeClassifier(random_state=1),
            "Random_Forest": RandomForestClassifier(random_state=1),
            "ExtraTree": ExtraTreesClassifier(random_state=1),
            "Gradient_Boost": GradientBoostingClassifier(random_state=1),
            "KNN": KNeighborsClassifier(),
            "Logistic_Regression": LogisticRegression(random_state=1),
            "Naive_Bayes": GaussianNB(),
            "SVM": SVC(random_state=1),
            "Rede_Neural": MLPClassifier(random_state=1, max_iter=800),
        }

    @property
    def MODEL_SELECT(self):
        """Atributo dinâmico que retorna o modelo selectionado."""
        return "Random_Forest"
