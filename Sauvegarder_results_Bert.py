import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(data_train['our rating'])
y_test = label_encoder.transform(data_test['our rating'])
y_dev = label_encoder.transform(data_dev['our rating'])


dataset_train = pd.read_csv("predictions/predictions_train_partfalse.csv")
scores_train = np.array(dataset_train['score_partially_false'])

dataset_dev = pd.read_csv("predictions/predictions_dev_partfalse.csv")
scores_dev = np.array(dataset_dev['score_partially_false'])

dataset_test = pd.read_csv("predictions/predictions_test_partfalse.csv")
scores_test = np.array(dataset_test['score_partially_false'])

df_train = pd.DataFrame({
    "score": scores_train,
    "label": y_train
})
df_train.to_csv("train_scores_partfalse_BERT.csv", index=False)

# Pour le jeu de test
df_test = pd.DataFrame({
    "score": scores_test,
    "label": y_test
})
df_test.to_csv("test_scores_partfalse_BERT.csv", index=False)

df_dev = pd.DataFrame({
    "score": scores_dev,
    "label": y_dev
})
df_dev.to_csv("dev_scores_partfalse_BERT.csv", index=False)

