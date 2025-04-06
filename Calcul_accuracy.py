import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from donnees.nettoyage import load_dataset, clean_dataset, add_columns
import numpy as np

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
data_train_hf = data_train[["full_text", "partially_false"]].rename(columns={"partially_false": "labels"})
data_dev_hf = data_dev[["full_text", "partially_false"]].rename(columns={"partially_false": "labels"})
data_test_hf = data_test[["full_text", "partially_false"]].rename(columns={"partially_false": "labels"})

y_train_true = np.array(data_train_hf['labels'])
y_dev_true = np.array(data_dev_hf['labels'])
y_test_true = np.array(data_test_hf['labels'])


dataset_train = pd.read_csv("predictions/predictions_train_partfalse.csv")
pred_train = np.array(dataset_train['pred_partially_false'])

dataset_dev = pd.read_csv("predictions/predictions_dev_partfalse.csv")
pred_dev = np.array(dataset_dev['pred_partially_false'])

dataset_test = pd.read_csv("predictions/predictions_test_partfalse.csv")
pred_test = np.array(dataset_test['pred_partially_false'])


print("Accuracy sur le jeu de train = ", accuracy_score(y_train_true, pred_train))
print("Accuracy sur le jeu de dev = ", accuracy_score(y_dev_true, pred_dev))
print("Accuracy sur le jeu de test = ", accuracy_score(y_test_true, pred_test))

print("Nombre de true présents dans train: ", np.sum(pred_train))
print("Nombre de true présents dans dev: ", np.sum(pred_dev))
print("Nombre de true présents dans test: ", np.sum(pred_test))
