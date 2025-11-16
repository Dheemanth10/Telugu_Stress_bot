import os
import json
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


DATA_JSONL = "final_dataset_complete.jsonl"
MODEL_NAME = "./indicbert_telugu_emotion"   
SAVE_DIR = "indicbert_final"                 
NUM_LABELS = 4
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
MAX_LEN = 128
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        text = str(self.texts[i])
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item['labels'] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


if not Path(DATA_JSONL).exists():
    raise FileNotFoundError(f"{DATA_JSONL} not found in current folder!")

rows = []
with open(DATA_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)

if 'text' not in df.columns or 'emotion' not in df.columns:
    raise ValueError("Dataset must contain 'text' and 'emotion' fields.")


le = LabelEncoder()
df['label_id'] = le.fit_transform(df['emotion'].astype(str))

label_map = {cls: int(i) for cls, i in zip(le.classes_, le.transform(le.classes_))}
print("Label map:", label_map)

train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['label_id'].tolist(),
    test_size=0.2,
    random_state=SEED,
    stratify=df['label_id'].tolist()
)


print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(DEVICE)


train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


optimizer = AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * EPOCHS
warmup_steps = max(1, int(0.1 * total_steps))
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    loop = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch} Training Loss: {avg_train_loss:.4f}")


    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Accuracy after Epoch {epoch}: {val_acc:.4f}")


    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(SAVE_DIR, exist_ok=True)
        print("Saving BEST model to:", SAVE_DIR)
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)

print("Training complete. Best val acc:", best_val_acc)
print("Model saved at:", SAVE_DIR)
print("Label encoder classes:", list(le.classes_))


label_map_clean = {k: int(v) for k, v in label_map.items()}

label_info = {
    "classes": list(le.classes_),
    "class_to_id": label_map_clean
}


save_path = os.path.join(SAVE_DIR, "label_encoder.json")

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(label_info, f, ensure_ascii=False, indent=2)

print("Label encoder saved at:", save_path)

