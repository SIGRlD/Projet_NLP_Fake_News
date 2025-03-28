import os  # Module pour interagir avec le système d'exploitation
import re  # Module pour les expressions régulières
import json  # Module pour manipuler les fichiers JSON
import torch  # Bibliothèque PyTorch pour le calcul tensoriel et l'apprentissage profond
import pandas as pd  # Bibliothèque pour manipuler des tableaux de données (DataFrame)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Charger le jeu de données
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Charger un jeu de données à partir d'un fichier csv, qui contient des échantillons de texte utilisés pour l'entraînement du tokenizer.

    Args :
        file_path (str) : Chemin du fichier du jeu de données.

    Returns :
        DataFrame : Un DataFrame contenant le jeu de données.
    """
    data_frame = pd.read_csv(file_path)

    return data_frame

def calculate_cosine_similarity(text: str, title: str) -> float:
    """
    Calculer la similarité cosinus entre deux textes.

    Args :
        text (str) : Le texte sur une ligne d'un data_frame.
        title (str) : Le titre sur la même ligne, lié au texte.

    Returns :
        float : Similarité cosinus entre les deux textes.
    """
    vectorizer = TfidfVectorizer().fit_transform([text, title])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0, 1]

# Nettoyer le jeu de données
def clean_dataset(data_frame_dirty: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyer un jeu de données en supprimant les valeurs manquantes et les doublons.

    Args :
        data_frame_dirty (DataFrame) : Un DataFrame contenant le jeu de données, pas encore nettoyé.

    Returns :
        DataFrame : Un DataFrame nettoyé.
    """
    data_frame_cleaned = data_frame_dirty.copy()

    # Retirer les espaces en trop
    data_frame_cleaned = data_frame_cleaned.replace("   ", " ")
    data_frame_cleaned = data_frame_cleaned.replace("  ", " ")

    # Supprimer les lignes où le texte ou le titre est manquant. On met à jour l'index des lignes.
    data_frame_cleaned = data_frame_cleaned.dropna(subset=["text", "title"]).reset_index(drop=True)

    # Supprimer les doublons
    data_frame_cleaned = data_frame_cleaned.drop_duplicates(subset=["text", "title"])

    # Je veux parcourir toutes les lignes du DataFrame
    i = 0
    while i < len(data_frame_cleaned):
        j = i + 1
        while j < len(data_frame_cleaned):
            # print("i: " + str(i) + " j: " + str(j))
            row_i = data_frame_cleaned.iloc[i]
            row_j = data_frame_cleaned.iloc[j]
            # On récupère le texte, le titre et la classe de la ligne
            text_i = row_i["text"]
            text_j = row_j["text"]
            title_i = row_i["title"]
            title_j = row_j["title"]
            rating_i = row_i["our rating"]
            rating_j = row_j["our rating"]

            # Si 2 lignes sont identiques
            if text_i == text_j and title_i == title_j:
                if rating_i == rating_j:
                    # On supprime la ligne j
                    data_frame_cleaned = data_frame_cleaned.drop(j).reset_index(drop=True)
                    j-=1
                else:
                    # On supprime les 2 lignes
                    data_frame_cleaned = data_frame_cleaned.drop(i).reset_index(drop=True)
                    data_frame_cleaned = data_frame_cleaned.drop(j).reset_index(drop=True)
                    break
            elif text_i == text_j or title_i == title_j:
                # On calcule la similarité cosinus entre le texte et le titre
                similarity_i = calculate_cosine_similarity(text_i, title_i)
                similarity_j = calculate_cosine_similarity(text_j, title_j)
                # On regarde quelle ligne a le plus de similarité entre le texte et le titre
                if similarity_i > similarity_j:
                    # On supprime la ligne j
                    data_frame_cleaned = data_frame_cleaned.drop(j).reset_index(drop=True)
                    j-=1
                else:
                    # On supprime la ligne i
                    data_frame_cleaned = data_frame_cleaned.drop(i).reset_index(drop=True)
                    break
            j+=1
        i+=1

    return data_frame_cleaned

def add_columns(data_frame: pd.DataFrame):
    """
    Ajouter les colonnes nécessaire pour utiliser les 4 modèles au DataFrame.

    Args :
        data_frame (DataFrame) : Un DataFrame.

    Returns :
        DataFrame : Un DataFrame avec les colonnes ajoutées.
    """
    data_frame["true"] = data_frame["our rating"].apply(lambda x: "TRUE" if x == "TRUE" else "rest")
    data_frame["false"] = data_frame["our rating"].apply(lambda x: "FALSE" if x == "FALSE" else "rest")
    data_frame["partially false"] = data_frame["our rating"].apply(lambda x: "partially false" if x == "partially false" else "rest")
    data_frame["other"] = data_frame["our rating"].apply(lambda x: "other" if x == "other" else "rest")

    return data_frame

if __name__ == "__main__":
    # Charger le jeu de données
    data_frame = load_dataset("Data/Task3_english_training.csv")

    print(data_frame.head())

    # On supprime l'id de la ligne
    data_frame = data_frame.drop(columns=["public_id"])

    # On nettoye le jeu de données
    data_frame = clean_dataset(data_frame)

    # On ajoute les colonnes dans le tableau pour les 4 modèles
    data_frame = add_columns(data_frame)

    print(data_frame.head())

