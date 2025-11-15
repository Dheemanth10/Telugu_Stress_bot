# test_final_model.py
# Run: source venv/bin/activate
#      python test_final_model.py

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "indicbert_final"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

# Load label map
with open(f"{MODEL_DIR}/label_encoder.json", "r", encoding="utf-8") as f:
    label_info = json.load(f)
class_to_id = label_info["class_to_id"]

id_to_class = {int(v): k for k, v in class_to_id.items()}

softmax = torch.nn.Softmax(dim=1)

def compound_from_confidence(emotion, confidence):
    if emotion == "normal":
        compound = 0.4 + 0.6 * confidence          # [0.4, 1.0]
    elif emotion == "depression":
        compound = -0.9 + 0.4 * (1 - confidence)   # [-0.9, -0.5]
    elif emotion == "stress":
        compound = -0.7 + 0.6 * (1 - confidence)   # [-0.7, -0.1]
    elif emotion == "anxiety":
        compound = -0.6 + 0.9 * (1 - confidence)   # [-0.6, 0.3]
    else:
        compound = 0.0
    # clamp
    return round(max(-1.0, min(1.0, compound)), 2)

def intensity_from_compound(compound):
    # intensity = round(((1 - compound) / 2) * 10) clamped to 1..10
    val = int(round(((1 - compound) / 2) * 10))
    return max(1, min(10, val))

print("Loaded model from:", MODEL_DIR)
print("Type a Telugu sentence and press Enter. Type 'exit' to quit.\n")

while True:
    text = input("Enter Telugu text (or 'exit'): ").strip()
    if text.lower() == "exit":
        print("Exiting.")
        break
    if text == "":
        print("Empty input — please type a Telugu sentence (or 'exit').\n")
        continue

    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

    # move tensors to device
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits  # shape (1, num_labels)
        probs = softmax(logits).cpu().numpy().flatten()
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])

    pred_label = id_to_class.get(pred_id, str(pred_id))
    compound = compound_from_confidence(pred_label, confidence)
    intensity = intensity_from_compound(compound)

    print(f"Predicted Emotion: {pred_label}")
    print(f"Intensity (1-10): {intensity}   |   Compound: {compound}\n")
