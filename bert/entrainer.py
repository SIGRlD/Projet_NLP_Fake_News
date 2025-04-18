import torch
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertForSequenceClassification, Trainer, TrainingArguments
from donnees.ajout_donnees import ajout_donnees
from donnees.nettoyage import load_dataset, clean_dataset, add_columns


def entrainer_binaire(chemin_train, chemin_dev, chemin_output, label, ajout_data=False):
    """
    Cette fonction permet d effectuer l entrainement d un modele BERT pour une classe au choix
    :param chemin_train: chemin vers le fichier des donnees d entrainement
    :param chemin_dev: chemin vers le fichier de dev
    :param chemin_output: chemin vers le dossier des outputs, mieux vaut specifier un nouveau dossier
    :param label: label pour le modele binaire, 0 si faux, 1 si autre, 2 si partiellement faux, 3 si vrai
    :param ajout_data: vrai si on reequilibre les classes pour avoir un dataset bien reparti entre les quatre labels
    :return:
    """
    # Reequilibrage des classes si specifie
    if ajout_data:
        data_train_hf = ajout_donnees(chemin_train)
    else:
        data_train = load_dataset(chemin_train)
        data_train = clean_dataset(data_train)
        data_train = add_columns(data_train)
        data_train_hf = data_train[["full_text", "our rating"]].rename(columns={"our rating": "labels"})

    #Recuperation du jeu de dev
    data_dev = load_dataset(chemin_dev)
    data_dev = clean_dataset(data_dev)
    data_dev = add_columns(data_dev)
    data_dev_hf = data_dev[["full_text", "our rating"]].rename(columns={"our rating": "labels"})

    # Transformation des labels en valeurs numeriques
    # false: 0, other: 1, partfalse: 2, true: 3
    label_order = ['false', 'other', 'partially false', 'true']
    label_encoder = LabelEncoder()
    label_encoder.fit(label_order)
    data_train_hf['labels'] = label_encoder.transform(data_train_hf['labels'])
    data_dev_hf['labels'] = label_encoder.transform(data_dev_hf['labels'])

    # On convertit en float pour le modele
    data_train_hf['labels'] = (data_train_hf['labels'] == label).astype(float)
    data_dev_hf['labels'] = (data_dev_hf['labels'] == label).astype(float)

    # On transforme les dataframe en objets dataset
    dataset_train = Dataset.from_pandas(data_train_hf)
    dataset_dev = Dataset.from_pandas(data_dev_hf)

    # Tokenizer et modele
    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # On definit la fonction de tokenization
    def tokenize(batch):
        return tokenizer(batch["full_text"], padding="max_length", truncation=True)

    # On tokenize nos datasets
    dataset_train = dataset_train.map(tokenize, batched=True)
    dataset_dev = dataset_dev.map(tokenize, batched=True)

    # On recupere le modele
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if label>0:
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1, device_map=device)
    else:
        model = DistilBertForSequenceClassification.from_pretrained(checkpoint, num_labels=1, device_map=device)

    # On entraine
    if label>0:
        args = TrainingArguments(
            output_dir="./modeles/checkpoints",  # Sauvegardes
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            report_to="none",
        )
    else:
        args = TrainingArguments(
            output_dir="./modeles/checkpoints",
            eval_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=10,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_steps=10,
            report_to="none",
        )

    # On initialise un objet trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
    )

    # On lance l entrainement et on sauvegarde le resultat
    trainer.train()
    model.save_pretrained(chemin_output)
    tokenizer.save_pretrained(chemin_output)
