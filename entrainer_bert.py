import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from donnees.nettoyage import load_dataset, clean_dataset, add_columns
import numpy as np

data_train = load_dataset("donnees/FakeNews_Task3_2022/Task3_train_dev/Task3_english_training.csv")
data_train = clean_dataset(data_train)
data_train = add_columns(data_train)
print(data_train.columns)
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

label_encoder = LabelEncoder()
data_train_hf['labels'] = label_encoder.fit_transform(data_train_hf['labels'])
data_dev_hf['labels'] = label_encoder.fit_transform(data_dev_hf['labels'])

dataset_train = Dataset.from_pandas(data_train_hf)
dataset_dev = Dataset.from_pandas(data_dev_hf)

# Tokenizer
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize(batch):
    return tokenizer(batch["full_text"], padding="max_length", truncation=True)

dataset_train = dataset_train.map(tokenize, batched=True)
dataset_dev = dataset_dev.map(tokenize, batched=True)

# Modèle
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4)

# Entraînement
args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset_train,
    eval_dataset=dataset_dev,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("./bert_model_all_class/")
tokenizer.save_pretrained("./bert_model_all_class/")

