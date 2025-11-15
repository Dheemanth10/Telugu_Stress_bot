# prepare_flan_data.py
import pandas as pd
import json

# Load dataset
df = pd.read_csv("translated_output.csv")

def parse_exclaim(s):
    """Convert comma-separated string into lowercase label set."""
    if pd.isna(s):
        return set()
    return set(x.strip().lower() for x in str(s).split(","))

def safe_float(x):
    """Safely convert to float, default to 0.0."""
    try:
        return float(x)
    except (ValueError, TypeError):
        return 0.0

rows = []

for i, r in df.iterrows():
    labels = parse_exclaim(r.get("status"))
    comp = safe_float(r.get("compound", 0))

    # Determine the main emotional class
    if any(lbl in labels for lbl in ["anxiety"]):
        main_label = "anxiety"
    elif any(lbl in labels for lbl in ["depression", "suicidal"]):
        main_label = "depression"
    elif any(lbl in labels for lbl in ["stress", "bipolar"]):
        main_label = "stress"
    else:
        main_label = "normal"

    # Skip invalid entries (no Telugu sentence)
    telugu_text = r.get("telugu_translation", "")
    if not isinstance(telugu_text, str) or not telugu_text.strip():
        print(f"⚠️ Skipping row {i}: missing Telugu translation")
        continue

    # Compose final row
    rows.append({
        "instruction": "Classify the emotional state of the Telugu sentence into one of: anxiety, depression, stress, or normal, and report the compound score.",
        "input": telugu_text.strip(),
        "output": f"Class: {main_label}, Compound: {comp:.2f}"
    })

# Save clean JSONL file
out_path = "telugu_stress_classes_compound.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Saved {len(rows)} examples to {out_path}")