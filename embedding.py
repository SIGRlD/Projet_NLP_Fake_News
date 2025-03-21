import numpy as np
import pandas as pd
import time
import torch
from nltk.tokenize import word_tokenize
from IPython.display import clear_output


class GloVeModel():
    def __init__(self, fichier: str):
        """
        Embedding GloVe pré-entrainé. 

        Paramètre
            fichier: chemin du fichier contenant les vecteurs pré-entrainés
        
        Source : https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
        """
        self.dico = {}
        self.index_to_key = []
        with open(fichier,'r') as f:
            for line in f:
                split_line = line.split()
                word = split_line[0]
                vecteur = np.array(split_line[1:],dtype=np.float32)
                self.dico[word] = vecteur
                self.index_to_key.append(word)
    
    def __getitem__(self, cle):
        if cle in self.index_to_key:
            return self.dico[cle]
        else:
            raise Exception(f"Key '{cle}' not present")


def embedding(mots: list, modele):
    """
    Fonction qui jetonise les mots. 
    Ignore les mots qui ne sont pas dans le vocabulaire. 

    Entrées
        mots: liste des mots à jetoniser n_mots
        modele: modèle à utiliser

    Sortie
        tenseur des jetons (n_mots, n_emb)
    """
    with torch.no_grad():
        taille = modele[modele.index_to_key[0]].shape[0]
        jetons = torch.zeros((1,taille))
        for m in mots:
            if m in modele.index_to_key:
                jetons = torch.row_stack((jetons,torch.from_numpy(modele[m].copy())))
        return jetons[1:,:]


def padding(jetons: list):
    """
    Fonction qui padde les séquences pour qu'elles aient toutes la même taille. 

    Entrée
        jetons: liste des jetons n_phrases * (n_mots_i, n_emb)

    Sortie
        tenseur des jetons paddés (n_phrases, max_mots, n_emb)
    """
    with torch.no_grad():
        taille_max = max([j.shape[0] for j in jetons])
        tenseur = []
        for j in jetons:
            manque = taille_max-j.shape[0]
            tenseur.append(torch.row_stack((j,torch.zeros((manque,j.shape[1]))))[None,:,:])
        return torch.cat(tenseur)


def tokeniser(corpus: pd.Series, modele, pad: bool = False):
    """
    Fonction qui jetonise un corpus. 
    Ignore les mots qui ne sont pas dans le vocabulaire. 

    Entrées
        corpus: données à jetoniser (n_phrases,)
        modele: modèle à utiliser
        pad: si on padde les séquences

    Sortie
        liste des jetons n_phrases * (n_mots_i, n_emb)
        ou tenseur des jetons (n_phrases, max_mots, n_emb)
    """
    tokens_all = []
    size = corpus.shape[0]
    somme = 0
    somme_temps = 0
    for i,x in enumerate(corpus.index):
        debut = time.perf_counter()
        mots = word_tokenize(corpus[x])
        if not isinstance(modele,GloVeModel):
            mots = [m if m!="." else "</s>" for m in mots]
        somme += len(mots)
        tokens = embedding(mots,modele)
        tokens_all.append(tokens)
        fin = time.perf_counter()
        somme_temps += fin-debut
        if (i+1)%50==0:
            clear_output(wait=True)
            temps = somme_temps/(i+1)
            restant = int((size-i-1)*temps)
            print("Progression : {}/{} ({:.1f}%) | Nombre de mots moyen par ligne : {:.1f} | ETA : {} m {} s".format(i+1,size,100*(i+1)/size,somme/(i+1),restant//60,restant%60))
    clear_output(wait=True)
    if pad:
        tenseur = padding(tokens_all)
        print("Progression : {}/{} ({:.1f}%) | Nombre de mots moyen par ligne : {:.1f} | Temps : {} m {} s".format(i+1,size,100*(i+1)/size,somme/(i+1),int(somme_temps)//60,int(somme_temps)%60))
        return tenseur
    else:
        print("Progression : {}/{} ({:.1f}%) | Nombre de mots moyen par ligne : {:.1f} | Temps : {} m {} s".format(i+1,size,100*(i+1)/size,somme/(i+1),int(somme_temps)//60,int(somme_temps)%60))
        return tokens_all
