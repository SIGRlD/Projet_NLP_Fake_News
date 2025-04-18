# Projet_NLP_Fake_News
Projet de traitement automatique des langues naturelles  
Nathan Beaujean, Antoine Beunas, Tom Daguerre, Yuhan Ménard-Tétreault

## Structure du répertoire
Ce répertoire contient tous les scripts nécessaires au projet.  
* **train.py** : script principal pour l'entrainement des modèles de classification binaire
* **predict.py** : script principal pour les prédictions des modèles de classification binaire
* **bert** : dossier contenant les scripts pour les modèles Bert
  * *entrainer.py* : fonction d'entrainement d'un modèle Bert
  * *predire.py* : fonction de prédictions d'un modèle Bert
* **donnees** : dossier contenant les données et des scripts de gestion des données 
  * *FakeNews_Task3_2022_V0* : dossier des données brutes
  * *glove.6B* : dossier des vecteurs pré-entrainés GloVe
  * *resultats* : dossier des résultats (prédictions) des modèles
  * *ajout_donnees.py* : fonction d'augmentation de données pour l'entrainement
  * *embedding.py* : fonctions d'embedding des données
  * *nettoyage.py* : fonctions de nettoyage des données
  * *utils.py* : fonctions utilitaires pour manipuler les données pré-entrainement
* **modeles** : dossier contenant les modèles entrainés
  * *checkpoints* : dossier des modèles Bert sauvegardés lors de l'entrainement
* **reseaux_neurones** : dossier contenant les scripts pour les modèles CNN, RNN et FFNN
  * *modeles.py* : classes et fonctions pour l'entrainement de réseaux de neurones
  * *modelisation.ipynb* : notebook pour l'entrainement et les prédictions de réseaux de neurones

### Modèles retenus
Suite à l'évaluation de différents types de modèles, les modèles retenus pour le résultat final sont les modèles **Bert**.  
Ainsi, les scripts liés aux réseaux de neurones CNN, RNN et FFNN sont *dépréciés*. 

## Installer les dépendances
Pour installer les librairies nécessaires à l'exécution des différents scripts, il suffit d'entrer la ligne de commande `pip install -r requirements.txt` dans un terminal. 

## Entrainer un modèle
Pour entrainer un modèle, entrer la ligne de commande `python train.py [parametres]` dans un terminal.  
Pour plus d'informations sur les paramètres, entrer la commande `python train.py --help`.  
Voici quelques exemples d'usage : 
* `python train.py --model=true`
* `python train.py --model=true --model_dir=./modeles/bert_true`
* `python train.py --model=true --model_dir=./modeles/bert_true --add_data`
Si aucun chemin de données est spécifié, les données sont supposées dans le dossier *donnees/FakeNews_Task3_2022_V0* portant leur nom original. 

## Obtenir des prédictions d'un modèle
Pour obtenir des prédictions d'un modèle, entrer la ligne de commande `python predict.py [parametres]` dans un terminal.  
Pour plus d'informations sur les paramètres, entrer la commande `python predict.py --help`.  
Voici quelques exemples d'usage : 
* `python predict.py --model=true --data_path=./donnees/FakeNews_Task3_2022_V0/English_data_test_release_with_rating.csv --dataset=test`
* `python predict.py --model=true --model_dir=./modeles/bert_true --data_path=./donnees/FakeNews_Task3_2022_V0/English_data_test_release_with_rating.csv --dataset=test`
Si aucun chemin de résultats est spécifié, les résultats sont enregistrés dans le dossier *"donnees/resultats"*. 
