import pandas as pd


def FakeNews_Task3_2022_V0(chemin: str, ensemble: str):
    """
    Fonction qui effectue le pré-traitement du jeu de données FakeNews_Task3_2022_V0. 

    Entrées
        chemin: chemin des données brutes
        ensemble: nom de l'ensemble de données (train, dev ou test)

    Sortie
        données pré-traitées
    """
    data = pd.read_csv(chemin)
    if ensemble=="train":
        assert data.shape==(900,4), "Les données brutes n'ont pas les bonnes dimensions..."
        assert (data.columns==["public_id", "text", "title", "our rating"]).all(), "Les données brutes n'ont pas les bonnes colonnes..."
        data = data[data.title.notna()].copy()
        data.drop_duplicates(["text","title"],keep="first",inplace=True)
        data["our_rating"] = data["our rating"].str.lower()
        data = data[~(data.public_id.isin(["0dcce357","3d66ab2a","77b92ae4","bac80f96"]))].copy()
        data.drop("our rating",axis=1,inplace=True)
        assert data.shape==(875,4) and (data.columns==["public_id", "text", "title", "our_rating"]).all(), "Il y a eu une erreur..."
    elif ensemble=="dev":
        assert data.shape==(364,4), "Les données brutes n'ont pas les bonnes dimensions..."
        assert (data.columns==["public_id", "text", "title", "our rating"]).all(), "Les données brutes n'ont pas les bonnes colonnes..."
        data = data[data.title.notna()].copy()
        data.drop_duplicates(["text","title"],inplace=True)
        data["our_rating"] = data["our rating"].str.lower()
        data.drop("our rating",axis=1,inplace=True)
        assert data.shape==(348,4) and (data.columns==["public_id", "text", "title", "our_rating"]).all(), "Il y a eu une erreur..."
    elif ensemble=="test":
        assert data.shape==(612,4), "Les données brutes n'ont pas les bonnes dimensions..."
        assert (data.columns==["ID", "text", "title", "our rating"]).all(), "Les données brutes n'ont pas les bonnes colonnes..."
        data["our_rating"] = data["our rating"].str.lower()
        data.drop("our rating",axis=1,inplace=True)
        assert data.shape==(612,4) and (data.columns==["ID", "text", "title", "our_rating"]).all(), "Il y a eu une erreur..."
    else:
        raise Exception("Ensemble invalide! Valeurs permises : 'train', 'dev', 'test'.")
    return data
