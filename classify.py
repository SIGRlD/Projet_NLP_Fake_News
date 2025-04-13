import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from kadv import K_Avocats
from textblob import TextBlob
import nltk
from donnees.nettoyage import load_dataset, clean_dataset, add_columns
from sklearn.ensemble import ExtraTreesClassifier

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
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity
def get_polarity(text):
    return TextBlob(text).sentiment.polarity

def avg_words_per_sentence(text):
    sentences = nltk.sent_tokenize(text)
    words = [len(nltk.word_tokenize(s)) for s in sentences]
    return sum(words) / len(sentences) if sentences else 0


subjectivity_train = np.array(data_train["full_text"].apply(get_subjectivity))
polarity_train = np.array(data_train["full_text"].apply(get_polarity))
len_train = np.array(data_train["full_text"].apply(avg_words_per_sentence))

subjectivity_dev = np.array(data_dev["full_text"].apply(get_subjectivity))
polarity_dev = np.array(data_dev["full_text"].apply(get_polarity))
len_dev = np.array(data_dev["full_text"].apply(avg_words_per_sentence))

subjectivity_test = np.array(data_test["full_text"].apply(get_subjectivity))
polarity_test = np.array(data_test["full_text"].apply(get_polarity))
len_test = np.array(data_test["full_text"].apply(avg_words_per_sentence))

print("JEU DE TRAIN")
#RECUPERER LES SCORES DU JEU D ENTRAINEMENT
dataset = pd.read_csv("scores/train_scores_false.csv")
scores_train_false = np.array(dataset['score'])

dataset = pd.read_csv("scores/train_scores_partfalse_BERT.csv")
scores_train_partfalse = np.array(dataset['score'])

dataset = pd.read_csv("scores/train_scores_other_BERT.csv")
scores_train_other = np.array(dataset['score'])

dataset = pd.read_csv("scores/train_scores_true_BERT.csv")
scores_train_true = np.array(dataset['score'])
labels = np.array(dataset['label'])

print(scores_train_true.shape)
scores_train_all = np.column_stack([scores_train_false, scores_train_partfalse,
                                    scores_train_other, scores_train_true, subjectivity_train, labels])
print(scores_train_all.shape)

print("JEU DE TEST")
#RECUPERER LES SCORES DU JEU DE TEST
dataset = pd.read_csv("scores/test_scores_false.csv")
scores_test_false = np.array(dataset['score'])

dataset = pd.read_csv("scores/test_scores_partfalse_BERT.csv")
scores_test_partfalse = np.array(dataset['score'])

dataset = pd.read_csv("scores/test_scores_other_BERT.csv")
scores_test_other = np.array(dataset['score'])

dataset = pd.read_csv("scores/test_scores_true_BERT.csv")
scores_test_true = np.array(dataset['score'])
labels_tests = np.array(dataset['label'])

print(scores_test_true.shape)
scores_test_all = np.column_stack([scores_test_false, scores_test_partfalse,
                                   scores_test_other, scores_test_true, subjectivity_test, labels_tests])
print(scores_test_all.shape)

print("JEU DE DEV")
#RECUPERER LES SCORES DU JEU DE DEV
dataset = pd.read_csv("scores/dev_scores_false.csv")
scores_dev_false = np.array(dataset['score'])

dataset = pd.read_csv("scores/dev_scores_partfalse_BERT.csv")
scores_dev_partfalse = np.array(dataset['score'])

dataset = pd.read_csv("scores/dev_scores_other_BERT.csv")
scores_dev_other = np.array(dataset['score'])

dataset = pd.read_csv("scores/dev_scores_true_BERT.csv")
scores_dev_true = np.array(dataset['score'])
labels_dev = np.array(dataset['label'])

print(scores_dev_true.shape)
scores_dev_all = np.column_stack([scores_dev_false, scores_dev_partfalse,
                                  scores_dev_other, scores_dev_true, subjectivity_dev, labels_dev])
print(scores_dev_all.shape)

X = scores_train_all[:, :5]
y = scores_train_all[:, 5]
# print(X.shape)
X_dev = scores_dev_all[:, :5]
y_dev = scores_dev_all[:, 5]
X_test = scores_test_all[:, :5]
y_test = scores_test_all[:, 5]

# distances = np.linalg.norm(X - X.mean(axis=0), axis=1)
# seuil = np.percentile(distances, 90)  # ou 80
# X = X[distances < seuil]
# y = y[distances < seuil]
#------------------------------------------------------------
# mask = np.zeros(len(X), dtype=bool)
#
# classes = np.unique(y)
#
# for c in classes:
#     indices = np.where(y == c)[0]
#     X_c = X[indices]
#     # Centre du cluster de classe c
#     center_c = X_c.mean(axis=0)
#     # Distance à ce centroïde
#     distances = np.linalg.norm(X_c - center_c, axis=1)
#     # Seuil : ici, on garde les 90% les plus proches
#     seuil = np.percentile(distances, 90)
#     conserves = distances < seuil
#     # Marquer les lignes à conserver
#     mask[indices[conserves]] = True
#
# # Appliquer le filtre
# X = X[mask]
# y = y[mask]


print("ARBRE DE CLASSIFICATION SIMPLE")
model = DecisionTreeClassifier()
model.fit(X, y)
pred_dev = model.predict(X_dev)
pred_train = model.predict(X)
pred_test = model.predict(X_test)

print("Accuracy sur train = ", accuracy_score(y, pred_train))
print("Accuracy sur dev = ", accuracy_score(y_dev, pred_dev))
print("Accuracy sur test = ", accuracy_score(y_test, pred_test))


print("FORET DE DECISION")
# model_f = RandomForestClassifier(
#     n_estimators=200,         # plus d’arbres = plus stable
#     max_depth=8,              # limite la complexité de chaque arbre
#     min_samples_split=10,     # évite les splits trop fins
#     min_samples_leaf=4,       # pareil
#     max_features='sqrt',      # sous-ensemble de features à chaque split (classique)
#     class_weight='balanced',  # gère le déséquilibre des classes
#     random_state=42
# )
model_f = RandomForestClassifier(n_estimators=100, random_state=42)
model_f.fit(X, y)
pred_dev = model_f.predict(X_dev)
pred_train = model_f.predict(X)
pred_test = model_f.predict(X_test)

print("Accuracy sur train = ", accuracy_score(y, pred_train))
print("Accuracy sur dev = ", accuracy_score(y_dev, pred_dev))
print("Accuracy sur test = ", accuracy_score(y_test, pred_test))
importances = model_f.feature_importances_
features = ["false", "other", "partfalse", "true", "sub"]
plt.barh(features, importances)
plt.title("Importance des features")
plt.show()

print("EXTRA TREE CLASSIFIER")
model_e = ExtraTreesClassifier(n_estimators=300, max_depth=10, random_state=42)
model_e.fit(X, y)
pred_dev = model_e.predict(X_dev)
pred_train = model_e.predict(X)
pred_test = model_e.predict(X_test)

print("Accuracy sur train = ", accuracy_score(y, pred_train))
print("Accuracy sur dev = ", accuracy_score(y_dev, pred_dev))
print("Accuracy sur test = ", accuracy_score(y_test, pred_test))

print("XGBCLASSIFIER")
model_x = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
    n_estimators=50,          # Réduire le nombre d’arbres
    max_depth=3,              # Limiter la profondeur
    learning_rate=0.05,       # Moins agressif
    subsample=0.7,            # Moins de données par arbre
    colsample_bytree=0.8,     # Moins de features
    reg_lambda=5,             # Régularisation L2
    reg_alpha=1               # Régularisation L1
)
model_x.fit(X, y)
pred_dev = model_x.predict(X_dev)
pred_train = model_x.predict(X)
pred_test = model_x.predict(X_test)

print("Accuracy sur train = ", accuracy_score(y, pred_train))
print("Accuracy sur dev = ", accuracy_score(y_dev, pred_dev))
print("Accuracy sur test = ", accuracy_score(y_test, pred_test))

print("MLP")
model_m = MLPClassifier(hidden_layer_sizes=(8, 16,), max_iter=750, random_state=42)
model_m.fit(X, y)
pred_dev = model_m.predict(X_dev)
pred_train = model_m.predict(X)
pred_test = model_m.predict(X_test)

print("Accuracy sur train = ", accuracy_score(y, pred_train))
print("Accuracy sur dev = ", accuracy_score(y_dev, pred_dev))
print("Accuracy sur test = ", accuracy_score(y_test, pred_test))

print("Logistic regression")

model_l = LogisticRegression(multi_class='multinomial', max_iter=1000)
model_l.fit(X, y)
pred_dev = model_l.predict(X_dev)
pred_train = model_l.predict(X)
pred_test = model_l.predict(X_test)

print("Accuracy sur train = ", accuracy_score(y, pred_train))
print("Accuracy sur dev = ", accuracy_score(y_dev, pred_dev))
print("Accuracy sur test = ", accuracy_score(y_test, pred_test))

print("K-AVOCATS")
model_A = K_Avocats(4, 0.005, seuil=0.8, n_iterations=0)
model_A.fit(X, y)
pred_dev = model_A.predict(X_dev)
pred_train = model_A.predict(X)
pred_test = model_A.predict(X_test)

print("Accuracy sur train = ", accuracy_score(y, pred_train))
print("Accuracy sur dev = ", accuracy_score(y_dev, pred_dev))
print("Accuracy sur test = ", accuracy_score(y_test, pred_test))
