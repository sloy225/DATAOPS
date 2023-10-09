import numpy as np
import urllib.request
import pandas as pd


from sklearn.tree import DecisionTreeClassifier
#url ='https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
#data = urllib.request.urlopen(url)

df = pd.read_csv("titanic.csv")

selected_columns = ["Sex", "Pclass", "Age", "Survived", "Fare", "Embarked"]
data_subset = df[selected_columns]
print(data_subset)
# Renommage la colonne "fare" en "price" et "Pclass" en "Class"
data_subset = data_subset.rename(columns={"fare": "price"})
data_subset = data_subset.rename(columns={"Pclass": "class"})
données_passager = data_subset.to_dict(orient="records")
# Conversion les données en format JSON
data_subset= data_subset.dropna()
données_passager = data_subset.to_json(orient="records")

# Enregistrement des données dans un fichier JSON
with open('passenger.json', 'w') as file:
    file.write(données_passager)


# Fonction pour charger les données depuis une URL
def requestUrl(url):
    try:
        data = pd.read_csv(url)
        return data
    except Exception as e:
        print(f"Erreur de chargement de données : {e}")
        return None

def extract_model(url):
    if url is not None:
        # Séparer les caractéristiques (X) de la cible (y)
        X = url.drop(columns=["survived"])
        y = url["survived"]

        # Créer un modèle, par exemple, un arbre de décision
        model = DecisionTreeClassifier()

        # Entraîner le modèle sur les données
        model.fit(X, y)

        return model
    else:
        return None


