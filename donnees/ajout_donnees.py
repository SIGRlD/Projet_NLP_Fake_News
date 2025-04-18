import numpy as np
import pandas as pd
from donnees.nettoyage import load_dataset, clean_dataset, add_columns

def ajout_donnees(chemin):
    data_train = load_dataset(chemin)
    data_train = clean_dataset(data_train)
    data_train = add_columns(data_train)

    # Exemple : créer un dataset HF à partir de tes pandas
    data_train_hf = data_train[["full_text", "our rating"]].rename(columns={"our rating": "labels"})

    #Bien pour other!!
    splits = {'train': 'train.csv', 'validation': 'dev.csv', 'test': 'test.csv'}
    add_train = pd.read_csv("hf://datasets/pszemraj/multi_fc/" + splits["train"])

    # On veut que toutes les classes soient réparties de cette meniere
    #false 0 other 1 partfalse 2 true 3
    ref_len = len(data_train_hf[data_train_hf['labels'] == "false"])
    true_add_len = np.minimum(ref_len-len(data_train_hf[data_train_hf['labels'] == "true"]),
                              len(add_train[add_train['label'] == "true"]))

    other_add_len = np.minimum(ref_len-len(data_train_hf[data_train_hf['labels'] == "other"]),
                               len(add_train[add_train['label'] == "fiction!"]))

    partfalse_add_len = np.minimum(ref_len-len(data_train_hf[data_train_hf['labels'] == "partially false"]),
                                   len(add_train[add_train['label'] == "half true"]))

    true_add = add_train[add_train['label'] == "true"]['claim'][:true_add_len]
    other_add = add_train[add_train['label'] == "fiction!"]['claim'][:other_add_len]
    partfalse_add = add_train[add_train['label'] == "half true"]['claim'][:partfalse_add_len]

    df_true = pd.DataFrame({'full_text': true_add, 'labels': 'true'})
    df_other = pd.DataFrame({'full_text': other_add, 'labels': 'other'})
    df_partfalse = pd.DataFrame({'full_text': partfalse_add, 'labels': 'partially false'})

    data_train_hf = pd.concat([data_train_hf, df_true, df_other, df_partfalse], ignore_index=True)
    data_train_hf.to_csv("./donnees/train_augmente.csv", index=False)
    return data_train_hf
