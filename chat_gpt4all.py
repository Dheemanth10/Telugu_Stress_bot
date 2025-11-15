# chat_gpt4all.py
# Run:
#   source venv/bin/activate
#   python chat_gpt4all.py
'''
import json
import random
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gpt4all import GPT4All

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
MODEL_DIR = "indicbert_final"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your local GGUF LLM model (no internet required)
LLAMA_PATH = "models/mistral-7b-instruct.Q4_0.gguf"   # <-- file must exist here


# -------------------------------------------------------------------
# LOAD CLASSIFIER
# -------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

with open(f"{MODEL_DIR}/label_encoder.json", "r", encoding="utf-8") as f:
    label_info = json.load(f)
class_to_id = label_info["class_to_id"]
id_to_class = {int(v): k for k, v in class_to_id.items()}

softmax = torch.nn.Softmax(dim=1)


# -------------------------------------------------------------------
# CLASSIFIER HELPERS
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# Remedies
# -------------------------------------------------------------------
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


# --------------------------
# GPT4ALL LOCAL OFFLINE LLM
# --------------------------
GPT4ALL_MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_0.gguf"   # filename
GPT4ALL_MODEL_DIR  = "models"                          # folder containing file

llm = GPT4All(
    model_name=GPT4ALL_MODEL_NAME,
    model_path=GPT4ALL_MODEL_DIR,
    allow_download=False,      # fully offline
    verbose=True
)



# -------------------------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# MAKE REPLY FROM GPT4ALL
# -------------------------------------------------------------------
def generate_reply(system, user_msg):
    prompt = f"{system}\n\nUser: {user_msg}\nAssistant:"
    output = llm.generate(prompt, max_tokens=200, temp=0.5)
    return output.strip()


# -------------------------------------------------------------------
# CLASSIFIER PIPELINE
# -------------------------------------------------------------------
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


# -------------------------------------------------------------------
# CHAT LOOP
# -------------------------------------------------------------------
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
'''

# chat_gpt4all_telugu.py
# Usage:
#   source venv/bin/activate
#   python chat_gpt4all_telugu.py

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gpt4all import GPT4All
import re

# -------------------------
# CONFIG
# -------------------------
MODEL_DIR = "indicbert_final"
MAX_LEN = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local model (put the .gguf in models/)
GPT4ALL_MODEL_NAME = "mistral-7b-instruct-v0.2.Q4_0.gguf"  # or qwen2.5... if you have it
GPT4ALL_MODEL_DIR  = "models"

# -------------------------
# Load classifier
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

with open(f"{MODEL_DIR}/label_encoder.json", "r", encoding="utf-8") as f:
    label_info = json.load(f)
class_to_id = label_info["class_to_id"]
id_to_class = {int(v): k for k, v in class_to_id.items()}
softmax = torch.nn.Softmax(dim=1)

# -------------------------
# Classifier helpers
# -------------------------
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

# -------------------------
# Remedies
# -------------------------
REMEDIES = {
    "normal": ["మీరు బాగున్నారు — అలాగే కొనసాగండి!"],
    "anxiety": ["4-4-4 లోతైన శ్వాస వ్యాయామం చేయండి.", "చిన్న నడకకు వెళ్లి వెనక్కి ఉండు."],
    "stress": ["పనిని చిన్న భాగాలుగా విడగొట్టండి.", "5 నిమిషాల విరామం తీసుకోండి."],
    "depression": ["మీ నమ్మకమైన వ్యక్తితో మాట్లాడండి.", "ప్రొఫెషనల్ సహాయం పొందండి."],
    "unknown": ["ఇంకాస్త వివరంగా చెప్పండి."]
}
ESCALATION_COMPOUND_THRESHOLD = -0.75

# -------------------------
# Load GPT4All (offline, cpu)
# -------------------------
llm = GPT4All(
    model_name=GPT4ALL_MODEL_NAME,
    model_path=GPT4ALL_MODEL_DIR,
    allow_download=False,
    verbose=False
)

# -------------------------
# Utility: detect if text is mostly Telugu
# -------------------------
TELUGU_RE = re.compile(r'[\u0C00-\u0C7F]')

def telugu_ratio(text: str) -> float:
    if not text:
        return 0.0
    tel = len(TELUGU_RE.findall(text))
    total = len(text)
    return tel / total if total > 0 else 0.0

# -------------------------
# System prompt: force Telugu-only
# -------------------------
def system_prompt(emotion, confidence, compound, intensity, remedies):
    remedy_text = "\n".join(f"- {r}" for r in remedies)
    esc = ""
    if emotion == "depression" and compound <= ESCALATION_COMPOUND_THRESHOLD:
        esc = (
            "\n⚠️ ఇది తీవ్రమైన భావోద్వేగ స్థితి. వినియోగదారునికి ప్రొఫెషనల్ సహాయాన్ని సలహా చేయండి.\n"
        )

    return f"""
మీరు పూర్తిగా **తెలుగులో మాత్రమే** స్పందించే, సహానుభూతితో మాట్లాడే AI అసిస్టెంట్.

వినియోగదారుని భావోద్వేగ విశ్లేషణ:
- భావం: {emotion}
- నమ్మకం: {round(confidence,3)}
- compound: {compound}
- తీవ్రత: {intensity}/10

సూచనలు:
{remedy_text}

{esc}

**గమనికలు (అత్యంత ముఖ్యమైనవి):**
1) మీరు 100% తెలుగులో మాత్రమే స్పందించాలి — ఒకటి కూడా ఇంగ్లీష్ పదం రాకూడదు.
2) ఎంత చిన్నదైనా స్పష్టం చేసి, మృదువుగా, సహానుభూతితో స్పందించు.
3) కనీసం 2 అమలు చేయదగిన సూచనలు ఇవ్వండి.
4) చివరగా ఒక చిన్న follow-up ప్రశ్న అడగండి.
"""

# -------------------------
# LLM generation (streaming) - returns text
# -------------------------
def generate_streamed(prompt: str, temp: float = 0.6, max_tokens: int = 250) -> str:
    out = ""
    for token in llm.generate(prompt, streaming=True, temp=temp, max_tokens=max_tokens):
        print(token, end="", flush=True)
        out += token
    print()  # newline after generation
    return out.strip()

# -------------------------
# Generate assistant reply; if not Telugu, auto-translate
# -------------------------
def generate_reply(system, user_msg):
    # 1) Ask LLM to reply (Telugu-forced in system prompt)
    prompt = f"{system}\n\nUser (తెలుగులో): {user_msg}\nAssistant (తెలుగులో):"
    print_endline = False
    print("", end="")  # ensure next prints join well
    reply = generate_streamed(prompt, temp=0.6, max_tokens=250)

    # 2) if reply is not sufficiently Telugu, ask the model to translate the reply into Telugu
    if telugu_ratio(reply) < 0.25:
        # Give explicit instruction to translate to Telugu only
        print("\n[Detected non-Telugu output — translating to Telugu...]\n")
        trans_prompt = (
            "ఈ క్రింది వాక్యాన్ని పూర్తిగా తెలుగులోకి అనువదించండి. ఒకటి కూడా ఇంగ్లీష్ పదం వాడవద్దు.\n\n"
            f"TEXT TO TRANSLATE:\n{reply}\n\nTRANSLATED (తెలుగులో בלבד):"
        )
        translated = generate_streamed(trans_prompt, temp=0.3, max_tokens=200)
        return translated
    return reply

# -------------------------
# Classify
# -------------------------
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

# -------------------------
# Chat loop
# -------------------------
print("\nTelugu Emotional AI — 100% offline. Type 'exit' to quit.\n")
while True:
    user = input("You: ").strip()
    if user.lower() == "exit":
        print("Bot: జాగ్రత్తగా ఉండండి! మళ్లీ కలుద్దాం.\n")
        break
    if user == "":
        print("Bot: ఖాళీ సందేశం — దయచేసి టైప్ చేయండి.\n")
        continue

    emotion, confidence = classify_text(user)
    compound = compound_from_confidence(emotion, confidence)
    intensity = intensity_from_compound(compound)
    remedies = REMEDIES.get(emotion, REMEDIES["unknown"])
    sys_p = system_prompt(emotion, confidence, compound, intensity, remedies)

    print("\n--- Analysis ---")
    print("Emotion:", emotion)
    print("Confidence:", confidence)
    print("Compound:", compound)
    print("Intensity:", intensity)
    print("----------------\n")

    print("Bot: ", end="")
    final_reply = generate_reply(sys_p, user)
    # final_reply already printed token-by-token; also available as string
