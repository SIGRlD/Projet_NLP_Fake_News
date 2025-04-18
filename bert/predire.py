import torch
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification
from donnees.nettoyage import load_dataset, clean_dataset, add_columns


def predire_binaire(chemin_entree, chemin_output, chemin_modele, label, ensemble):
    """
    Sauvegarde un csv de predictions dans un csv
    :param chemin_entree: chemin du fichier d entree
    :param chemin_output: chemin du dossier de l output
    :param chemin_modele: chemin du dossier contenant le modele
    :param label: label pour le modele binaire, 0 si faux, 1 si autre, 2 si partiellement faux, 3 si vrai
    :param ensemble nom de l'ensemble de données (train, dev, test, ...)
    :return: sauvegarde les predictions
    """
    data = load_dataset(chemin_entree)
    if "labels" not in data.columns and "our rating" not in data.columns:
        print("Dataset non conforme, il doit avoir une colonne 'labels' ou 'our rating'")
        return 2

    # Si labels n est pas dans les colonnes, alors le fichier a besoin d etre pretraite
    if "labels" not in data.columns:
        data = clean_dataset(data)
        data = add_columns(data)
        # Exemple : créer un dataset HF à partir de tes pandas
        data = data[["full_text", "our rating"]].rename(columns={"our rating": "labels"})
        label_order = ['false', 'other', 'partially false', 'true']
        label_encoder = LabelEncoder()
        label_encoder.fit(label_order)
        data['labels'] = label_encoder.transform(data['labels'])

    # Gestion d erreurs pour le format du dataset
    if "full_text" not in data.columns:
        print("Dataset non conforme, il doit avoir la colonne 'full_text'")
        return 1

    # Chemins du modele et de la sortie
    model_dir = chemin_modele
    output_csv = chemin_output+f"{ensemble}_scores_{label_order[label]}.csv"
    batch_size = 16

    # On utilise cuda si possible, on definit le modlee a utiliser ainsi que le tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if label>0:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1, device_map=device)
    else:
        model = DistilBertForSequenceClassification.from_pretrained(model_dir, num_labels=1, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    # On recupere les textes
    texts = data["full_text"].tolist()

    # On predit en gardant les labels et scores de cote
    scores = []
    for i in tqdm(range(0, len(texts), batch_size)):
        # On recupere un batch de texts qu on tokenize
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

        # On calcule les outputs du modele et on recupere les scores
        with torch.no_grad():
            outputs = model(**inputs)
            if label>0:
                probs = torch.sigmoid(outputs.logits)
            else:
                probs = outputs.logits

        scores.extend(probs[:, 0].cpu().numpy())  # Score pour la classe

    # On sauvegarde le csv des predictions
    data["scores"] = scores
    out_csv = data[["scores", "labels"]]
    out_csv.to_csv(output_csv, index=False)

    print(f"Prédictions enregistrées dans {output_csv}")
