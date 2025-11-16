import json
import re
import pandas as pd

input_file = "telugu_stress_classes_compound.jsonl"
output_jsonl = "final_dataset.jsonl"
output_csv = "final_dataset.csv"
class_pattern = r"Class:\s*(\w+)"
compound_pattern = r"Compound:\s*(-?\d+\.\d+|-?\d+)"

def compound_to_intensity(compound):
    """
    Convert compound (-1 to 1) to intensity (1 to 10).
    -1 → 10, +1 → 1
    """
    intensity = ((-compound + 1) / 2) * 9 + 1
    return int(round(max(1, min(10, intensity))))  

records = []

with open(input_file, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, start=1):
        item = json.loads(line)

        text = item.get("input", "")

        output = item.get("output", "")
        cls = re.search(class_pattern, output)
        emotion = cls.group(1).lower() if cls else "normal"

        cmp = re.search(compound_pattern, output)
        compound = float(cmp.group(1)) if cmp else 0.0

        intensity = compound_to_intensity(compound)

        records.append({
            "id": idx,
            "text": text,
            "emotion": emotion,
            "compound": compound,
            "intensity": intensity
        })

with open(output_jsonl, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

df = pd.DataFrame(records)
df.to_csv(output_csv, index=False)

print("Dataset conversion complete!")
print("✔ Saved:", output_jsonl)
print("✔ Saved:", output_csv)
print("✔ Total samples:", len(records))
