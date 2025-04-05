from sklearn.metrics import accuracy_score
import pandas as pd
from embedding import TfIdf
from gensim.models import KeyedVectors
from donnees.nettoyage import load_dataset, clean_dataset, add_columns
from donnees.utils import FakeNewsDataset, ajuster_canaux
from embedding import GloVeModel, tokeniser
from sklearn.preprocessing import LabelEncoder
import numpy as np
from classifier_other_and_true import classifier_models

# Entrainement
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

print(f"Entrainement : {data_train.shape[0]}, {data_train.shape[1]} | Validation : {data_dev.shape[0]} | Test : {data_test.shape[0]}")
embedding = TfIdf(data_train.full_text)
print(embedding.X.shape)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(data_train['our rating'])
y_test = label_encoder.transform(data_test['our rating'])
y_dev = label_encoder.transform(data_dev['our rating'])

y_train_other = []
for pred in y_train:
  if pred == 1:
    y_train_other.append(1)
  else:
    y_train_other.append(0)

y_test_other = []
for pred in y_test:
  if pred == 1:
    y_test_other.append(1)
  else:
    y_test_other.append(0)
y_test_other = np.array(y_test_other)

y_dev_other = []
for pred in y_dev:
  if pred == 3:
    y_dev_other.append(1)
  else:
    y_dev_other.append(0)
y_dev_true = np.array(y_dev_other)

classifieur = classifier_models(embedding.X, y_train_other)
embedding_tests = embedding.embedding_newdata(data_test.full_text)
pred = classifieur.predict_labels(embedding_tests)
scores = classifieur.predict_scores(embedding_tests)
# print(scores)

print("Accuracy sur le jeu de test = ", accuracy_score(y_test_other, pred))
scores_train = classifieur.predict_scores(embedding.X)

df_train = pd.DataFrame({
    "score": scores_train,
    "label": y_train
})
df_train.to_csv("train_scores_other.csv", index=False)

# Pour le jeu de test
df_test = pd.DataFrame({
    "score": scores,
    "label": y_test
})
df_test.to_csv("test_scores_other.csv", index=False)


embedding_dev = embedding.embedding_newdata(data_dev.full_text)
scores_dev = classifieur.predict_scores(embedding_dev)

df_dev = pd.DataFrame({
    "score": scores_dev,
    "label": y_dev
})
df_dev.to_csv("dev_scores_true.csv", index=False)

pred = classifieur.predict_labels(embedding_dev)
print("Accuracy sur le jeu de dev = ", accuracy_score(y_dev_true, pred))
