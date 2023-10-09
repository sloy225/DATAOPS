import numpy as np
import urllib.request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("titanic.csv")

def extract_model(data):
    if data is not None:
        # Séparer les caractéristiques (X) de la cible (y)
        X = data.drop(columns=["Survived"])
        y = data["Survived"]

        # Créer un modèle, par exemple, un arbre de décision
        Modele = DecisionTreeClassifier()

        # Entraîner le modèle sur les données
        Modele.fit(X, y)

        return Modele
    else:
        return None


extract_model(df)