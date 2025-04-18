import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay


# Modeles

class CNN(nn.Module):
    def __init__(self, input_size: int, in_channels: int, out_channels1: int, out_channels2: int, out_channels3: int, out_channels4: int, kernel_size: int, p_dropout, device: str):
        """
        Modèle CNN. 
        4 conv1D + 2 maxPool1D + 1 couche pleinement connectée. 
        Pour les séquences de texte de longueur fixe. 

        Paramètres
            input_size: taille des entrées
            in_channels: nombre de canaux d'entrée
            out_channels1: nombre de canaux de sortie (conv1)
            out_channels2: nombre de canaux de sortie (conv2)
            out_channels3: nombre de canaux de sortie (conv3)
            out_channels4: nombre de canaux de sortie (conv4)
            kernel_size: taille du filtre convolutif
            p_dropout: probabilité de dropout (float ou tuple)
            device: 'cpu', 'cuda', 'mps', ...
        """
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.out_channels3 = out_channels3
        self.out_channels4 = out_channels4
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        if isinstance(p_dropout,tuple):
            self.dropout = True
        elif p_dropout>0:
            self.dropout = True
        else:
            self.dropout = False
        self.device = device

        self.conv1 = nn.Conv1d(in_channels, out_channels1, kernel_size, padding=kernel_size//2, device=device)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels1, out_channels2, kernel_size, padding=kernel_size//2, device=device)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool1d(2,2)
        if isinstance(p_dropout,tuple):
            self.dropout1 = nn.Dropout1d(p_dropout[0])
        elif p_dropout>0:
            self.dropout1 = nn.Dropout1d(p_dropout)
        self.conv3 = nn.Conv1d(out_channels2, out_channels3, kernel_size, padding=kernel_size//2, device=device)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv1d(out_channels3, out_channels4, kernel_size, padding=kernel_size//2, device=device)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool1d(2,2)
        if isinstance(p_dropout,tuple):
            self.dropout2 = nn.Dropout1d(p_dropout[1])
        elif p_dropout>0:
            self.dropout2 = nn.Dropout1d(p_dropout)
        self.flat = nn.Flatten()
        self.sortie = nn.Linear(input_size*out_channels4//4, 1, device=device)
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor):
        """
        Propagation avant pour l'entrainement. 

        Entrée
            X: données d'entrée (n_phrases, max_mots, n_emb)
        
        Sortie
            sortie du modèle
        """
        scores = self.conv1(X)
        scores = self.relu1(scores)
        scores = self.conv2(scores)
        scores = self.relu2(scores)
        scores = self.maxpool1(scores)
        if self.dropout:
            scores = self.dropout1(scores)
        scores = self.conv3(scores)
        scores = self.relu3(scores)
        scores = self.conv4(scores)
        scores = self.relu4(scores)
        scores = self.maxpool2(scores)
        if self.dropout:
            scores = self.dropout2(scores)
        scores = self.flat(scores)
        scores = self.sortie(scores)
        return self.sig(scores)
    
    def predict(self, X: list, seuil: float = 0.5):
        """
        Fonction qui effectue une prédiction. 

        Entrée
            X: données à utiliser (n_phrases, max_mots, n_emb)
            seuil: seuil de la classe positive (p>seuil -> 1)
        
        Sortie
            prédictions du modèle (n_phrases,)
        """
        with torch.no_grad():
            pred = self.predict_proba(X)
            return torch.greater(pred,seuil).type(torch.int)
    
    def predict_proba(self, X: list):
        """
        Fonction qui effectue une prédiction (probabilités). 

        Entrée
            X: données à utiliser (n_phrases, max_mots, n_emb)
        
        Sortie
            prédictions (probabilités) du modèle (n_phrases,)
        """
        with torch.no_grad():
            return self.forward(X.to(self.device)).flatten().to("cpu")


class CNNRNN(nn.Module):
    def __init__(self, input_size: int, in_channels: int, out_channels1: int, out_channels2: int, kernel_size: int, hidden_size: int, p_dropout, device: str):
        """
        Modèle hybride CNN-RNN modifié. 
        Source originale : https://www.sciencedirect.com/science/article/pii/S2667096820300070. 
        Pour les séquences de texte de longueur fixe. 

        Paramètres
            input_size: taille des entrées
            in_channels: nombre de canaux d'entrée
            out_channels1: nombre de canaux de sortie (conv1)
            out_channels2: nombre de canaux de sortie (conv2)
            kernel_size: taille du filtre convolutif
            hidden_size: taille de la couche cachée (LSTM)
            p_dropout: probabilité de dropout (float ou tuple)
            device: 'cpu', 'cuda', 'mps', ...
        """
        super(CNNRNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels1 = out_channels1
        self.out_channels2 = out_channels2
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        self.p_dropout = p_dropout
        self.device = device

        self.conv1 = nn.Conv1d(in_channels, out_channels1, kernel_size, padding=kernel_size//2, device=device)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels1, out_channels2, kernel_size, padding=kernel_size//2, device=device)
        self.relu2 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2,2)
        if p_dropout>0:
            self.dropout = nn.Dropout1d(p_dropout)
        self.lstm = nn.LSTM(input_size//2, hidden_size, num_layers=2, batch_first=True, dropout=p_dropout, device=device)
        self.sortie = nn.Linear(hidden_size, 1, device=device)
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor):
        """
        Propagation avant pour l'entrainement. 

        Entrée
            X: données d'entrée (n_phrases, max_mots, n_emb)
        
        Sortie
            sortie du modèle
        """
        scores = self.conv1(X)
        scores = self.relu1(scores)
        scores = self.conv2(scores)
        scores = self.relu2(scores)
        scores = self.maxpool(scores)
        if self.p_dropout>0:
            scores = self.dropout(scores)
        scores = self.lstm(scores)[0][:,-1,:]
        scores = self.sortie(scores)
        return self.sig(scores)
    
    def predict(self, X: list, seuil: float = 0.5):
        """
        Fonction qui effectue une prédiction. 

        Entrée
            X: données à utiliser (n_phrases, max_mots, n_emb)
            seuil: seuil de la classe positive (p>seuil -> 1)
        
        Sortie
            prédictions du modèle (n_phrases,)
        """
        with torch.no_grad():
            pred = self.predict_proba(X)
            return torch.greater(pred,seuil).type(torch.int)
    
    def predict_proba(self, X: list):
        """
        Fonction qui effectue une prédiction (probabilités). 

        Entrée
            X: données à utiliser (n_phrases, max_mots, n_emb)
        
        Sortie
            prédictions (probabilités) du modèle (n_phrases,)
        """
        with torch.no_grad():
            return self.forward(X.to(self.device)).flatten().to("cpu")


class FFNN(nn.Module):
    def __init__(self, input_size: int, in_size: int, hidden_size1: int, hidden_size2: int, hidden_size3: int, p_dropout, device: str):
        """
        Modèle feedforward. 
        1 couche entrée + 3 couches cachées + 1 couche sortie. 
        Pour données au niveau "phrases". 

        Paramètres
            input_size: taille des entrées
            in_size: taille de la couche d'entrée
            hidden_size1: taille de la couche cachée 1
            hidden_size2: taille de la couche cachée 2
            hidden_size3: taille de la couche cachée 3
            p_dropout: probabilité de dropout (float ou tuple)
            device: 'cpu', 'cuda', 'mps', ...
        """
        super(FFNN, self).__init__()
        self.in_size = in_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.p_dropout = p_dropout
        if isinstance(p_dropout,tuple):
            self.dropout = True
        elif p_dropout>0:
            self.dropout = True
        else:
            self.dropout = False
        self.device = device

        self.entree = nn.Linear(input_size, in_size, device=device)
        self.relu1 = nn.ReLU()
        if isinstance(p_dropout,tuple):
            self.dropout1 = nn.Dropout(p_dropout[0])
        elif p_dropout>0:
            self.dropout1 = nn.Dropout(p_dropout)
        self.cachee1 = nn.Linear(in_size, hidden_size1, device=device)
        self.relu2 = nn.ReLU()
        if isinstance(p_dropout,tuple):
            self.dropout2 = nn.Dropout(p_dropout[1])
        elif p_dropout>0:
            self.dropout2 = nn.Dropout(p_dropout)
        self.cachee2 = nn.Linear(hidden_size1, hidden_size2, device=device)
        self.relu3 = nn.ReLU()
        if isinstance(p_dropout,tuple):
            self.dropout3 = nn.Dropout(p_dropout[2])
        elif p_dropout>0:
            self.dropout3 = nn.Dropout(p_dropout)
        self.cachee3 = nn.Linear(hidden_size2, hidden_size3, device=device)
        self.relu4 = nn.ReLU()
        if isinstance(p_dropout,tuple):
            self.dropout4 = nn.Dropout(p_dropout[3])
        elif p_dropout>0:
            self.dropout4 = nn.Dropout(p_dropout)
        self.sortie = nn.Linear(hidden_size3, 1, device=device)
        self.sig = nn.Sigmoid()
    
    def forward(self, X: torch.Tensor):
        """
        Propagation avant pour l'entrainement. 

        Entrée
            X: données d'entrée (n_phrases, taille_vocab)
        
        Sortie
            sortie du modèle
        """
        scores = self.entree(X)
        scores = self.relu1(scores)
        if self.dropout:
            scores = self.dropout1(scores)
        scores = self.cachee1(scores)
        scores = self.relu2(scores)
        if self.dropout:
            scores = self.dropout2(scores)
        scores = self.cachee2(scores)
        scores = self.relu3(scores)
        if self.dropout:
            scores = self.dropout3(scores)
        scores = self.cachee3(scores)
        scores = self.relu4(scores)
        if self.dropout:
            scores = self.dropout4(scores)
        scores = self.sortie(scores)
        return self.sig(scores)
    
    def predict(self, X: list, seuil: float = 0.5):
        """
        Fonction qui effectue une prédiction. 

        Entrée
            X: données à utiliser (n_phrases, taille_vocab)
            seuil: seuil de la classe positive (p>seuil -> 1)
        
        Sortie
            prédictions du modèle (n_phrases,)
        """
        with torch.no_grad():
            pred = self.predict_proba(X)
            return torch.greater(pred,seuil).type(torch.int)
    
    def predict_proba(self, X: list):
        """
        Fonction qui effectue une prédiction (probabilités). 

        Entrée
            X: données à utiliser (n_phrases, taille_vocab)
        
        Sortie
            prédictions (probabilités) du modèle (n_phrases,)
        """
        with torch.no_grad():
            return self.forward(X.to(self.device)).flatten().to("cpu")


# Fonctions

def train_seq_fix(net, optimizer, max_epochs: int, Xy_train: Dataset, Xy_val: Dataset, taille_batch: int, melanger: bool, device: str, verbose: int = 1):
    """
    Fonction qui entraine un réseau de neurones. 
    Pour les séquences de texte de longueur fixe. 

    Entrées
        net: modèle
        optimizer: optimiseur
        max_epochs: nombre maximum d'epochs
        Xy_train: données d'entrainement (n_phrases_t, max_mots, n_emb)
        Xy_val: données de validation (n_phrases_v, max_mots, n_emb)
        taille_batch: taille des batchs d'entrainement
        melanger: si on mélange les données d'entrainement
        device: 'cpu', 'cuda', 'mps', ...
        verbose: fréquence d'affichage de la progression
    """
    train_dataloader = DataLoader(Xy_train,taille_batch,melanger)
    loss_fn = nn.BCELoss().to(device)
    erreurs = {"train": [], "valid": []}
    somme_temps = 0
    
    for epoch in range(1,max_epochs+1):
        debut = time.perf_counter()
        train_losses = []
        
        for X,y in train_dataloader:
            optimizer.zero_grad()
            pred_train = net(X.to(device))
            train_loss = loss_fn(pred_train,y[:,None].to(device)).mean()
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())
        erreurs["train"].append(np.mean(train_losses))

        with torch.no_grad():
            pred_val = net(Xy_val.X.to(device))
            val_loss = loss_fn(pred_val,Xy_val.y[:,None].to(device)).mean()
        erreurs["valid"].append(val_loss.item())

        fin = time.perf_counter()
        somme_temps += fin-debut
        if epoch%verbose==0:
            clear_output(wait=True)
            temps = somme_temps/epoch
            restant = int((max_epochs-epoch)*temps)
            print("Epochs : {} | Perte train : {:.4f} | Perte valid : {:.4f} | ETA : {} m {} s".format(epoch,erreurs["train"][-1],erreurs["valid"][-1],restant//60,restant%60))

    clear_output(wait=True)
    print("Epochs : {} | Perte train : {:.4f} | Perte valid : {:.4f} | Temps : {} m {} s".format(epoch,erreurs["train"][-1],erreurs["valid"][-1],int(somme_temps)//60,int(somme_temps)%60))
    plt.plot(range(1,max_epochs+1),erreurs["train"],label="Entrainement")
    plt.plot(range(1,max_epochs+1),erreurs["valid"],label="Validation")
    plt.legend()
    plt.show()


def evaluation(y_true: torch.Tensor, y_pred: torch.Tensor, ensemble: str, normaliser: str = None, multi: bool = False):
    """
    Fonction qui affiche les résultats du modèle (métriques et matrice de confusion). 
    Pour la classification binaire. 

    Entrées
        y_true: tenseur des étiquettes (n_phrases,)
        y_pred: tenseur des prédictions (n_phrases,)
        ensemble: nom de l'ensemble de données (train, dev, test, ...)
        normaliser: paramètre pour la matrice de confusion (None, 'true', 'pred' ou 'all')
        multi: s'il y a plus de 2 classes
    """
    moyenne = "micro" if multi else "binary"
    print("Justesse {} : {:.2f}%".format(ensemble,accuracy_score(y_true,y_pred)*100))
    print("Précision {} : {:.2f}%".format(ensemble,precision_score(y_true,y_pred,average=moyenne)*100))
    print("Rappel {} : {:.2f}%".format(ensemble,recall_score(y_true,y_pred,average=moyenne)*100))
    print("Score F1 {} : {:.2f}%".format(ensemble,f1_score(y_true,y_pred,average=moyenne)*100))
    ConfusionMatrixDisplay.from_predictions(y_true,y_pred,normalize=normaliser)
    plt.title(f"Matrice de confusion - Données {ensemble}")
    plt.show()
    plt.close()
