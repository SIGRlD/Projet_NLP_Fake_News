import numpy as np
import pandas as pd
from donnees.nettoyage import load_dataset, clean_dataset, add_columns

data_train = load_dataset("donnees/FakeNews_Task3_2022/Task3_train_dev/Task3_english_training.csv")
data_train = clean_dataset(data_train)
data_train = add_columns(data_train)
# Validation
data_dev = load_dataset("donnees/FakeNews_Task3_2022/Task3_train_dev/Task3_english_dev.csv")
data_dev = clean_dataset(data_dev)
data_dev = add_columns(data_dev)
# Test
data_test = load_dataset("donnees/FakeNews_Task3_2022/Task3_Test/English_data_test_release_with_rating.csv")
data_test = clean_dataset(data_test)
data_test = add_columns(data_test)

# Exemple : créer un dataset HF à partir de tes pandas
data_train_hf = data_train[["full_text", "our rating"]].rename(columns={"our rating": "labels"})
data_dev_hf = data_dev[["full_text", "our rating"]].rename(columns={"our rating": "labels"})
data_test_hf = data_test[["full_text", "our rating"]].rename(columns={"our rating": "labels"})

#Bien pour other!!
splits = {'train': 'train.csv', 'validation': 'dev.csv', 'test': 'test.csv'}
other_add_train = pd.read_csv("hf://datasets/pszemraj/multi_fc/" + splits["train"])
other_add_dev = pd.read_csv("hf://datasets/pszemraj/multi_fc/" + splits["validation"])
other_add_test = pd.read_csv("hf://datasets/pszemraj/multi_fc/" + splits["test"])

# print(other_add_train[other_add_train['label'] == "fiction!"].head(10)['claim'])
# print(other_add_train[other_add_train['label'] == "half true"].head(10)['claim'])
# print(other_add_train[other_add_train['label'] == "true"].head(10)['claim'])
print(len(data_train_hf[data_train_hf['labels'] == "false"]))
print(len(data_train_hf[data_train_hf['labels'] == "true"]))
print(len(data_train_hf[data_train_hf['labels'] == "partially false"]))
print(len(data_train_hf[data_train_hf['labels'] == "other"]))

# On veut que toutes les classes soient réparties de cette meniere
#false 0 other 1 partfalse 2 true 3
ref_len = len(data_train_hf[data_train_hf['labels'] == "false"])
true_add_len = np.minimum(ref_len-len(data_train_hf[data_train_hf['labels'] == "true"]),
                          len(other_add_train[other_add_train['label'] == "true"]))

other_add_len = np.minimum(ref_len-len(data_train_hf[data_train_hf['labels'] == "true"]),
                           len(other_add_train[other_add_train['label'] == "fiction!"]))

partfalse_add_len = np.minimum(ref_len-len(data_train_hf[data_train_hf['labels'] == "true"]),
                               len(other_add_train[other_add_train['label'] == "half true"]))

true_add = other_add_train[other_add_train['label'] == "true"]['claim'][:true_add_len]
other_add = other_add_train[other_add_train['label'] == "fiction!"]['claim'][:other_add_len]
partfalse_add = other_add_train[other_add_train['label'] == "half true"]['claim'][:partfalse_add_len]

df_true = pd.DataFrame({'full_text': true_add, 'labels': 'true'})
df_other = pd.DataFrame({'full_text': other_add, 'labels': 'other'})
df_partfalse = pd.DataFrame({'full_text': partfalse_add, 'labels': 'partially false'})

data_train_hf = pd.concat([data_train_hf, df_true, df_other, df_partfalse], ignore_index=True)
data_train_hf.to_csv("train_augmente.csv", index=False)
