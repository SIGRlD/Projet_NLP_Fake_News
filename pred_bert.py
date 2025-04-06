import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from donnees.nettoyage import load_dataset, clean_dataset, add_columns
import argparse
from datasets import Dataset

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

# ========== ARGUMENTS ========== #
model_dir = "bert_model_partfalse/"
output_csv = "predictions_train_partfalse.csv"
batch_size = 16

# ========== CHARGEMENT DU MODELE ========== #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model.eval()
#
# # ========== CHARGEMENT DES DONNEES ========== #
#
# texts = data_train_hf["full_text"].tolist()
#
# # ========== PRÉDICTION ========== #
# scores, labels = [], []
# for i in tqdm(range(0, len(texts), batch_size)):
#     batch_texts = texts[i:i+batch_size]
#     inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1)
#
#     scores.extend(probs[:, 1].cpu().numpy())           # Score pour la classe "true"
#     labels.extend(torch.argmax(probs, dim=1).cpu().numpy())  # 0 ou 1
#
# # ========== SAUVEGARDE ========== #
# data_train_hf["score_partially_false"] = scores
# data_train_hf["pred_partially_false"] = labels
# data_train_hf.to_csv(output_csv, index=False)
#
# print(f"✅ Prédictions enregistrées dans {output_csv}")
#-------------------------------------------------------------------------------------------------
# output_csv = "predictions_dev_partfalse.csv"
# texts = data_dev_hf["full_text"].tolist()
#
# # ========== PRÉDICTION ========== #
# scores, labels = [], []
# for i in tqdm(range(0, len(texts), batch_size)):
#     batch_texts = texts[i:i+batch_size]
#     inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1)
#
#     scores.extend(probs[:, 1].cpu().numpy())           # Score pour la classe "true"
#     labels.extend(torch.argmax(probs, dim=1).cpu().numpy())  # 0 ou 1
#
# # ========== SAUVEGARDE ========== #
# data_dev_hf["score_partially_false"] = scores
# data_dev_hf["pred_partially_false"] = labels
# data_dev_hf.to_csv(output_csv, index=False)
#
# print(f"Prédictions enregistrées dans {output_csv}")
#-----------------------------------------------------------------------------------------
output_csv = "predictions_test_partfalse.csv"
texts = data_test_hf["full_text"].tolist()

# ========== PRÉDICTION ========== #
scores, labels = [], []
for i in tqdm(range(0, len(texts), batch_size)):
    batch_texts = texts[i:i+batch_size]
    inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    scores.extend(probs[:, 1].cpu().numpy())           # Score pour la classe "true"
    labels.extend(torch.argmax(probs, dim=1).cpu().numpy())  # 0 ou 1

# ========== SAUVEGARDE ========== #
data_test_hf["score_partially_false"] = scores
data_test_hf["pred_partially_false"] = labels
data_test_hf.to_csv(output_csv, index=False)

print(f"Prédictions enregistrées dans {output_csv}")