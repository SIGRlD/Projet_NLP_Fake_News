import os  # Module pour interagir avec le système d'exploitation
import re  # Module pour les expressions régulières
import json  # Module pour manipuler les fichiers JSON
import torch  # Bibliothèque PyTorch pour le calcul tensoriel et l'apprentissage profond
import pandas as pd  # Bibliothèque pour manipuler des tableaux de données (DataFrame)
from donnees.nettoyage import load_dataset, clean_dataset, add_columns

if __name__ == "__main__":
    # Charger le jeu de données
    data_frame = load_dataset("Data/Task3_english_training.csv")

    print(data_frame.head())

    # On nettoye le jeu de données
    data_frame = clean_dataset(data_frame)

    # On ajoute les colonnes dans le tableau pour les 4 modèles
    data_frame = add_columns(data_frame)

    print(data_frame.head())
