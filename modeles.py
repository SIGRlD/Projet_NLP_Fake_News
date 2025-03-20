import numpy as np
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay


# Modeles

class LSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, device: str):
        """
        Modèle LSTM simple. 
        1 couche LSTM + 1 couche pleinement connectée. 
        Pour les séquences de texte à longueur variable. 

        Paramètres
            input_size: taille des entrées
            hidden_size: taille de la couche cachée
            device: 'cpu', 'cuda', 'mps', ...
        """
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, dropout=0.0, bidirectional=False, proj_size=0, device=device)
        self.sortie = nn.Linear(hidden_size, 1, bias=True, device=device)
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor):
        """
        Propagation avant pour l'entrainement. 

        Entrée
            X: une donnée d'entrée (n_mots_i, n_emb)
        
        Sortie
            sortie du modèle
        """
        scores = self.lstm(X)[0][-1,:]
        scores = self.sortie(scores)
        return self.sig(scores)
    
    def predict(self, X: list, seuil: float = 0.5):
        """
        Fonction qui effectue une prédiction. 

        Entrée
            X: données à utiliser n_phrases * (n_mots_i, n_emb)
            seuil: seuil de la classe positive (p>seuil -> 1)
        
        Sortie
            prédictions du modèle (n_phrases,)
        """
        with torch.no_grad():
            pred = self.predict_proba(X)
            return torch.greater(pred,seuil).type(torch.int32)
    
    def predict_proba(self, X: list):
        """
        Fonction qui effectue une prédiction (probabilités). 

        Entrée
            X: données à utiliser n_phrases * (n_mots_i, n_emb)
        
        Sortie
            prédictions (probabilités) du modèle (n_phrases,)
        """
        with torch.no_grad():
            pred = [self.forward(x.to(self.device)).item() for x in X]
            return torch.tensor(pred)


# Fonctions

def train_seq_var(net, optimizer, max_epochs: int, X_train: list, y_train: torch.Tensor, X_val: list, y_val: torch.Tensor, device: str, verbose: int = 10):
    """
    Fonction qui entraine un réseau de neurones. 
    Pour les séquences de texte à longueur variable. 

    Entrées
        net: modèle
        optimizer: optimiseur
        max_epochs: nombre maximum d'epochs
        X_train: données d'entrainement n_phrases_t * (n_mots_t_i, n_emb)
        y_train: étiquettes d'entrainement (n_phrases_t,)
        X_val: données de validation n_phrases_v * (n_mots_v_i, n_emb)
        y_val: étiquettes de validation (n_phrases_v,)
        device: 'cpu', 'cuda', 'mps', ...
        verbose: fréquence d'affichage de la progression
    """
    loss_fn = nn.BCELoss().to(device)
    erreurs = {"train": [], "dev": []}
    somme_temps = 0
    
    for epoch in range(1,max_epochs+1):
        debut = time.perf_counter()
        train_losses = []
        val_losses = []
        
        for i in range(len(X_train)):
            optimizer.zero_grad()
            pred_train = net(X_train[i].to(device))
            target_train = torch.tensor([y_train[i]]).to(device)
            train_loss = loss_fn(pred_train,target_train).mean()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
        erreurs["train"].append(np.mean(train_losses))

        for i in range(len(X_val)):
            with torch.no_grad():
                pred_val = net(X_val[i].to(device))
                target_val = torch.tensor([y_val[i]]).to(device)
                val_loss = loss_fn(pred_val,target_val).mean()
                val_losses.append(val_loss.item())
        erreurs["dev"].append(np.mean(val_losses))

        fin = time.perf_counter()
        somme_temps += fin-debut
        if epoch%verbose==0:
            clear_output(wait=True)
            temps = somme_temps/epoch
            restant = int((max_epochs-epoch)*temps)
            print("Epochs : {} | Perte train : {:.4f} | Perte dev : {:.4f} | ETA : {} m {} s".format(epoch,erreurs["train"][-1],erreurs["dev"][-1],restant//60,restant%60))

    clear_output(wait=True)
    print("Epochs : {} | Perte train : {:.4f} | Perte dev : {:.4f} | Temps : {} m {} s".format(epoch,erreurs["train"][-1],erreurs["dev"][-1],int(somme_temps)//60,int(somme_temps)%60))
    plt.plot(erreurs["train"],label="Entrainement")
    plt.plot(erreurs["dev"],label="Validation")
    plt.legend()
    plt.show()

def evaluation(y_true: torch.Tensor, y_pred: torch.Tensor, ensemble: str, normaliser: str = None):
    """
    Fonction qui affiche les résultats du modèle (métriques et matrice de confusion). 
    Pour la classification binaire. 

    Entrées
        y_true: tenseur des étiquettes (n_phrases,)
        y_pred: tenseur des prédictions (n_phrases,)
        ensemble: nom de l'ensemble de données (train, dev, test, ...)
        normaliser: paramètre pour la matrice de confusion (None, 'true', 'pred' ou 'all')
    """
    print("Justesse {} : {:.2f}%".format(ensemble,accuracy_score(y_true,y_pred)*100))
    print("Précision {} : {:.2f}%".format(ensemble,precision_score(y_true,y_pred)*100))
    print("Rappel {} : {:.2f}%".format(ensemble,recall_score(y_true,y_pred)*100))
    print("Score F1 {} : {:.2f}%".format(ensemble,f1_score(y_true,y_pred)*100))
    ConfusionMatrixDisplay.from_predictions(y_true,y_pred,normalize=normaliser)
    plt.title(f"Matrice de confusion - Données {ensemble}")
    plt.show()
    plt.close()
