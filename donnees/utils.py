import torch
from torch.utils.data import Dataset


class FakeNewsDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X.clone()
        self.y = y.clone()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def ajuster(X: torch.Tensor, nb_canaux: int):
    """
    Fonction qui ajuste le nombre de canaux d'un tenseur. 

    Entrées
        X: tenseur (i, j, k)
        nb_canaux: nombre de canaux désirés
    
    Sortie
        X ajusté (i, nb_canaux, k)
    """
    with torch.no_grad():
        manque = nb_canaux-X.shape[1]
        if manque>0:
            return torch.cat((X,torch.zeros((X.shape[0],manque,X.shape[2]))),1)
        elif manque<0:
            return X[:,:manque,:].clone()
        else:
            return X.clone()
