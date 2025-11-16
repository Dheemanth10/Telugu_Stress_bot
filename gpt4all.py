import json
import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gpt4all import GPT4All

MODEL_DIR = "indicbert_final"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LLAMA_PATH = "models/mistral-7b-instruct.Q4_0.gguf"    

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

with open(f"{MODEL_DIR}/label_encoder.json", "r", encoding="utf-8") as f:
    label_info = json.load(f)
class_to_id = label_info["class_to_id"]
id_to_class = {int(v): k for k, v in class_to_id.items()}

softmax = torch.nn.Softmax(dim=1)

def compound_from_confidence(emotion, confidence):
    if emotion == "normal":
        compound = 0.4 + 0.6 * confidence
    elif emotion == "depression":
        compound = -0.9 + 0.4 * (1 - confidence)
    elif emotion == "stress":
        compound = -0.7 + 0.6 * (1 - confidence)
    elif emotion == "anxiety":
        compound = -0.6 + 0.9 * (1 - confidence)
    else:
        compound = 0.0
    return round(max(-1.0, min(1.0, compound)), 2)


def intensity_from_compound(compound):
    val = int(round(((1 - compound) / 2) * 10))
    return max(1, min(10, val))

REMEDIES = {
    "normal": ["మీరు బాగున్నారు — అలాగే కొనసాగండి!"],
    "anxiety": [
        "4-4-4 డీప్ శ్వాస వ్యాయామం చేయండి.",
        "చిన్న నడక కు వెళ్లండి లేదా నీళ్లు తాగండి."
    ],
    "stress": [
        "పని ను చిన్న భాగాలుగా విడగొట్టండి.",
        "5 నిమిషాల విరామం తీసుకోండి."
    ],
    "depression": [
        "మీరు నమ్మే వ్యక్తితో మాట్లాడండి.",
        "సాధ్యమైతే ప్రొఫెషనల్ సహాయం పొందండి."
    ],
    "unknown": ["దయచేసి కొంచెం వివరంగా చెప్పండి."]
}

ESCALATION_COMPOUND_THRESHOLD = -0.75

GPT4ALL_MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_0.gguf"   
GPT4ALL_MODEL_DIR  = "models"                        

llm = GPT4All(
    model_name=GPT4ALL_MODEL_NAME,
    model_path=GPT4ALL_MODEL_DIR,
    allow_download=False,      
    verbose=True
)
def system_prompt(emotion, confidence, compound, intensity, remedies):
    remedy_text = "\n".join(f"- {r}" for r in remedies)

    esc = ""
    if emotion == "depression" and compound <= ESCALATION_COMPOUND_THRESHOLD:
        esc = (
            "\n⚠️ ఇది ప్రమాదకర స్థాయి భావం. "
            "వినియోగదారునికి నమ్మకమైన వ్యక్తితో మాట్లాడమని లేదా ప్రొఫెషనల్‌ను సంప్రదించమని మృదువుగా చెప్పండి.\n"
        )

    return f"""
మీరు ఒక సహానుభూతితో స్పందించే తెలుగులో మాట్లాడే AI అసిస్టెంట్.

వినియోగదారుని భావోద్వేగ విశ్లేషణ:
- భావం: {emotion}
- నమ్మకం: {round(confidence,3)}
- compound: {compound}
- తీవ్రత (1-10): {intensity}

ఉపయోగకరమైన సూచనలు:
{remedy_text}

{esc}

సూచనలు:
- చాలా మృదువుగా, అర్థమయ్యేలా, ధైర్యం చెప్పేలా మాట్లాడండి.
- కనీసం 2 ప్రాక్టికల్ సలహాలు ఇవ్వండి.
- చివర్లో ఒక చిన్న follow-up ప్రశ్న అడగండి.
- తెలుగులో మాత్రమే స్పందించండి.
"""

def generate_reply(system, user_msg):
    prompt = f"{system}\n\nUser: {user_msg}\nAssistant:"
    output = llm.generate(prompt, max_tokens=200, temp=0.5)
    return output.strip()

def classify_text(text):
    encoded = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

    with torch.no_grad():
        logits = model(**encoded).logits
        probs = softmax(logits).cpu().numpy().flatten()

    pred_id = int(np.argmax(probs))
    emotion = id_to_class[pred_id]
    confidence = float(probs[pred_id])

    return emotion, confidence
print("\nTelugu Emotional AI (IndicBERT + GPT4All) — 100% Offline")
print("Type 'exit' to quit.\n")

while True:
    user = input("You: ").strip()

    if user.lower() == "exit":
        print("Bot: జాగ్రత్తగా ఉండండి! ఎప్పుడైనా తిరిగి రండి.\n")
        break

    if user == "":
        print("Bot: ఖాళీ సందేశం వచ్చింది, దయచేసి ఏదైనా చెప్పండి.\n")
        continue

    # classify user message
    emotion, confidence = classify_text(user)
    compound = compound_from_confidence(emotion, confidence)
    intensity = intensity_from_compound(compound)
    remedies = REMEDIES.get(emotion, REMEDIES["unknown"])

    # system instruction for LLM
    sys_p = system_prompt(emotion, confidence, compound, intensity, remedies)

    # generate LLM reply
    reply = generate_reply(sys_p, user)

    # show classifier debug
    print("\n--- Analysis ---")
    print("Emotion:", emotion)
    print("Confidence:", confidence)
    print("Compound:", compound)
    print("Intensity:", intensity)
    print("----------------\n")

    print("Bot:", reply, "\n")
