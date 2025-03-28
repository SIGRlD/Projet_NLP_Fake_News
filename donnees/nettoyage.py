import pandas as pd  # Bibliothèque pour manipuler des tableaux de données (DataFrame)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

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
    data_frame_cleaned = data_frame_dirty[["title","text","our rating"]].copy()

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

    data_frame_cleaned["our rating"] = data_frame_cleaned["our rating"].str.lower()
    data_frame_cleaned["full_text"] = data_frame_cleaned.title+" "+data_frame_cleaned.text
    # for i in data_frame_cleaned.index:
    #     data_frame_cleaned["nb_mots"] = len(word_tokenize(data_frame_cleaned.full_text[i]))
    # print(data_frame_cleaned[data_frame_cleaned.nb_mots>3000].shape[0])
    # data_frame_cleaned = data_frame_cleaned[data_frame_cleaned.nb_mots<=3000].copy()
    # data_frame_cleaned.drop(columns=["full_text","nb_mots"],inplace=True)
    # data_frame_cleaned.reset_index(drop=True,inplace=True)

    return data_frame_cleaned

def add_columns(data_frame: pd.DataFrame):
    """
    Ajouter les colonnes nécessaire pour utiliser les 4 modèles au DataFrame.

    Args :
        data_frame (DataFrame) : Un DataFrame.

    Returns :
        DataFrame : Un DataFrame avec les colonnes ajoutées.
    """
    data_frame["true"] = data_frame["our rating"].apply(lambda x: 1 if x == "true" else 0)
    data_frame["false"] = data_frame["our rating"].apply(lambda x: 1 if x == "false" else 0)
    data_frame["partially_false"] = data_frame["our rating"].apply(lambda x: 1 if x == "partially false" else 0)
    data_frame["other"] = data_frame["our rating"].apply(lambda x: 1 if x == "other" else 0)

    return data_frame
