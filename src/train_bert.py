import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import mlflow
import os

# 1. LOAD DATA
print("📂 Loading data...")
train = pd.read_csv('data/processed/train.csv')
val = pd.read_csv('data/processed/val.csv')
test = pd.read_csv('data/processed/test.csv')
print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

# Pastiin semua teks jadi string & gak kosong
train['text'] = train['text'].astype(str).str.strip()
val['text']   = val['text'].astype(str).str.strip() 
test['text']  = test['text'].astype(str).str.strip()

# Buang teks kosong (kalau ada)
train = train[train['text'] != '']
val   = val[val['text'] != '']
test  = test[test['text'] != '']

print(f"Setelah bersihin: Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

# 2. TOKENIZER & DATASET
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class ReviewDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=128,
        )
        self.labels = labels.tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train['text'], train['label'])
val_dataset = ReviewDataset(val['text'], val['label'])
test_dataset = ReviewDataset(test['text'], test['label'])
print("✅ Tokenization selesai")

# 3. MODEL
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)

# 4. METRICS
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': acc, 'f1': f1}

# 5. TRAINING ARGS & TRAINER
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 6. TRAINING
print("🚀 Mulai training IndoBERT...")
mlflow.set_experiment("sentiment-bert")

with mlflow.start_run(run_name="indobert-local"):
    trainer.train()
    print("✅ Training selesai!")

    # Evaluasi test
    print("\n📊 Evaluasi di test set...")
    test_result = trainer.evaluate(test_dataset)
    mlflow.log_metrics({
        "test_accuracy": test_result['eval_accuracy'],
        "test_f1": test_result['eval_f1'],
    })
    print(f"Test Accuracy: {test_result['eval_accuracy']:.4f}")
    print(f"Test F1: {test_result['eval_f1']:.4f}")

    # Classification report
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    print("\n" + classification_report(
        test['label'], preds, target_names=['Negatif', 'Positif']
    ))

    # Simpan model
    os.makedirs('models/indobert', exist_ok=True)
    model.save_pretrained('models/indobert')
    tokenizer.save_pretrained('models/indobert')
    print("💾 Model disimpan di models/indobert/")

    mlflow.pytorch.log_model(model, "model")
    print(f"📁 Run ID: {mlflow.active_run().info.run_id}")

print("\n🎉 Done!")