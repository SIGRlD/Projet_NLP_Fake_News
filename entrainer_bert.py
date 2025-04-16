import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from donnees.nettoyage import load_dataset, clean_dataset, add_columns
import numpy as np

data_dev = load_dataset("donnees/FakeNews_Task3_2022/Task3_train_dev/Task3_english_dev.csv")
data_dev = clean_dataset(data_dev)
data_dev = add_columns(data_dev)
data_dev_hf = data_dev[["full_text", "our rating"]].rename(columns={"other": "labels"})

data_train_hf = pd.read_csv("train_augmente.csv")
label_encoder = LabelEncoder()
data_train_hf['labels'] = label_encoder.fit_transform(data_train_hf['labels'])
data_dev_hf['labels'] = label_encoder.fit_transform(data_dev_hf['labels'])
#false 0 other 1 partfalse 2 true 3
data_train_hf['labels'] = (data_train_hf['labels'] == 3).astype(float)
data_dev_hf['labels'] = (data_dev_hf['labels'] == 3).astype(float)

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
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1)

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
model.save_pretrained("./bert_model_true_equilibre/")
tokenizer.save_pretrained("./bert_model_true_equilibre/")

