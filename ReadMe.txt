PROJET NLP - Tom Daguerre, Nathan Beaujan, Antoine Beunas, Yuhan Ménard-Tétreault

Ce projet vise à classifier un jeu de données en 4 classes: true, false, partially false et other
Dans ce readme nous spécifierons les différentes commandes pour exécuter notre projet.

1 - Ajout de données
	Lancer le fichier ajout.py avec en argument le chemin vers le fichier train des données à augmenter
	Cela aura pour effet de produire un csv où les 4 classes ont exactement le même nombre de données

2 - Entrainer un modele BERT
	Lancer le fichier train.py avec l'argument "true", "false", "partially false", ou "other" pour lancer un entrainement binaire
	ou Lancer avec l'argument "all" pour entrainer un modèle sur les quatre classes
	L'entrainement peut être long (entre 1 et 2 heures par modèle) les modèles entrainés vous ont été envoyés pour passer outre cette étape
	Le fichier en sortie de l'ajout correspond aux attentes de format pour l'entrainement

3 - Predire sur de nouvelles données
	Lancer le fichier predict.py avec comme arguments le chemin vers votre csv de données et le chemin où les prédictions doivent s'enregistrer
	PS: Le csv doit soit avoir au moins les colonnes "full_text" et "labels" 
					soit avoir au moins les colonnes "full_text" et "our rating"

4 - Visualiser l'accuracy et le F1-Score et la représentation
	Parler notebook

Pour effectuer les opérations 3 - 4 - 5 d'un seul coup
	Lancer le fichier main.py avec comme arguments le chemin vers votre csv de données
	PS: Le csv doit soit avoir au moins les colonnes "full_text" et "labels" 
					soit avoir au moins les colonnes "full_text" et "our rating"