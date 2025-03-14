import pandas as pd
import numpy as np


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
        assert (data.columns==["public_id","text","title","our rating"]).all(), "Les données brutes n'ont pas les bonnes colonnes..."
        data = data[data.title.notna()].copy()
        data.drop_duplicates(["text","title"],keep="first",inplace=True)
        data["our_rating"] = data["our rating"].str.lower()
        data = data[~(data.public_id.isin(["0dcce357","3d66ab2a","77b92ae4","bac80f96"]))].copy()
        data.drop("our rating",axis=1,inplace=True)
        data.reset_index(drop=True,inplace=True)
        assert data.shape==(875,4) and (data.columns==["public_id","text","title","our_rating"]).all(), "Il y a eu une erreur..."
    elif ensemble=="dev":
        assert data.shape==(364,4), "Les données brutes n'ont pas les bonnes dimensions..."
        assert (data.columns==["public_id","text","title","our rating"]).all(), "Les données brutes n'ont pas les bonnes colonnes..."
        data = data[data.title.notna()].copy()
        data.drop_duplicates(["text","title"],inplace=True)
        data["our_rating"] = data["our rating"].str.lower()
        data.drop("our rating",axis=1,inplace=True)
        data.reset_index(drop=True,inplace=True)
        assert data.shape==(348,4) and (data.columns==["public_id","text","title","our_rating"]).all(), "Il y a eu une erreur..."
    elif ensemble=="test":
        assert data.shape==(612,4), "Les données brutes n'ont pas les bonnes dimensions..."
        assert (data.columns==["ID","text","title","our rating"]).all(), "Les données brutes n'ont pas les bonnes colonnes..."
        data["our_rating"] = data["our rating"].str.lower()
        data.drop("our rating",axis=1,inplace=True)
        assert data.shape==(612,4) and (data.columns==["ID","text","title","our_rating"]).all(), "Il y a eu une erreur..."
    else:
        raise Exception("Ensemble invalide! Valeurs permises : 'train', 'dev', 'test'.")
    return data


def Fake_Real_news(chemin_fake: str, chemin_true: str):
    """
    Fonction qui effectue le pré-traitement du jeu de données Fake_Real_news. 

    Entrées
        chemin_fake: chemin des données brutes (Fake)
        chemin_true: chemin des données brutes (True)

    Sortie
        données pré-traitées
    """
    # Donnees Fake
    data_fake = pd.read_csv(chemin_fake)
    assert data_fake.shape==(23481,4), "Les données brutes n'ont pas les bonnes dimensions..."
    assert (data_fake.columns==["title","text","subject","date"]).all(), "Les données brutes n'ont pas les bonnes colonnes..."
    data_fake.drop_duplicates(inplace=True)
    data_fake.drop_duplicates(["text","title"],keep="first",inplace=True)
    data_fake = data_fake[~(data_fake.text.isin([" ","  "]))].copy()
    data_fake.drop(18933,inplace=True)
    data_fake.drop(["subject","date"],axis=1,inplace=True)
    data_fake["label"] = "false"
    
    # Donnees True
    data_true = pd.read_csv(chemin_true)
    assert data_true.shape==(21417,4), "Les données brutes n'ont pas les bonnes dimensions..."
    assert (data_true.columns==["title","text","subject","date"]).all(), "Les données brutes n'ont pas les bonnes colonnes..."
    data_true.drop_duplicates(inplace=True)
    data_true.drop_duplicates(["text","title"],keep="first",inplace=True)
    data_true = data_true[~(data_true.text.isin([" "]))].copy()
    data_true.drop(["subject","date"],axis=1,inplace=True)
    data_true["label"] = "true"
    
    # Donnees full
    data = pd.concat((data_fake,data_true),ignore_index=True)
    assert data.shape==(38657,3) and (data.columns==["title","text","label"]).all(), "Il y a eu une erreur..."
    return data


def Veritas(chemin_v2: str, chemin_v2_1: str, chemin_v4: str, chemin_v4_1: str):
    """
    Fonction qui effectue le pré-traitement du jeu de données Fake_Real_news. 

    Entrées
        chemin_v2: chemin des données brutes (v2.0)
        chemin_v2_1: chemin des données brutes (v2.1)
        chemin_v4: chemin des données brutes (v4.0)
        chemin_v4_1: chemin des données brutes (v4.1)

    Sortie
        données pré-traitées
    """
    # Veritas v2
    data_v2 = pd.read_csv(chemin_v2,sep="\t",index_col=0)
    data_v21 = pd.read_csv(chemin_v2_1,sep="\t",index_col=0)
    assert data_v2.shape==(5107,8) and data_v21.shape==(4348,8), "Les données brutes n'ont pas les bonnes dimensions..."
    var2 = ["claim","claim_label","tags","claim_source_domain","claim_source_url"]
    data_v2 = data_v2[var2].rename(columns={"claim_source_domain": "source_domain","claim_source_url": "source_url"})
    data_v2.drop_duplicates(inplace=True)
    data_v21 = data_v21[var2].rename(columns={"claim_source_domain": "source_domain","claim_source_url": "source_url"})
    veritas_v2 = pd.concat((data_v2,data_v21),ignore_index=True)
    veritas_v2 = veritas_v2[veritas_v2.claim.notna()].copy()
    veritas_v2["claim"] = np.where(veritas_v2.claim.str.startswith("Claim: "),veritas_v2.claim.str[7:],veritas_v2.claim)
    veritas_v2["claim_label"] = veritas_v2.claim_label.str.lower()
    veritas_v2.drop_duplicates(["claim","claim_label"],inplace=True)
    veritas_v2["claim_label"] = np.select([veritas_v2.claim_label=="mtrue",veritas_v2.claim_label=="mfalse"],["mostly true","mostly false"],veritas_v2.claim_label)

    # Veritas v4
    data_v4 = pd.read_csv(chemin_v4,sep="\t",index_col=0)
    data_v41 = pd.read_csv(chemin_v4_1,sep="\t",index_col=0)
    assert data_v4.shape==(704,14) and data_v41.shape==(738,14), "Les données brutes n'ont pas les bonnes dimensions..."
    var4 = ["claim","verdict","a_tags","o_domain","o_url"]
    data_v4 = data_v4[var4].rename(columns={"verdict": "claim_label","a_tags": "tags","o_domain": "source_domain","o_url": "source_url"})
    data_v41 = data_v41[var4].rename(columns={"verdict": "claim_label","a_tags": "tags","o_domain": "source_domain","o_url": "source_url"})
    veritas_v4 = pd.concat((data_v4,data_v41),ignore_index=True)
    veritas_v4 = veritas_v4[veritas_v4.claim.notna()].copy()
    veritas_v4["claim"] = np.where(veritas_v4.claim.str.startswith("Claim: "),veritas_v4.claim.str[7:],veritas_v4.claim)
    veritas_v4["claim_label"] = veritas_v4.claim_label.str.lower()
    veritas_v4.drop_duplicates(["claim","claim_label"],inplace=True)
    veritas_v4["claim_label"] = np.select([veritas_v4.claim_label=="true.",veritas_v4.claim_label=="false.",veritas_v4.claim_label=="true (but not for the reason you think)"],["true","false","true"],veritas_v4.claim_label)

    # Veritas full
    data = pd.concat((veritas_v2,veritas_v4),ignore_index=True)
    data.drop_duplicates(["claim","claim_label"],inplace=True)
    data.drop(["source_domain","source_url"],axis=1,inplace=True)
    data.reset_index(drop=True,inplace=True)
    assert data.shape==(9329,3) and (data.columns==["claim","claim_label","tags"]).all(), "Il y a eu une erreur..."
    return data
