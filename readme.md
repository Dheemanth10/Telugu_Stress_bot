# Telugu Stress Detection Bot  
A machine-learning based system designed to detect **stress levels in Telugu text**, including Telugu–English code-mixed sentences.  
This repository includes data processing scripts, model-training pipelines, intensity converters, and testing utilities.

---

## 🚀 Features
- Telugu text preprocessing (cleaning, normalization, tokenization)
- Stress intensity classification
- Sentiment-compound → intensity conversion
- Complete training pipeline
- Evaluation & inference scripts
- GPT4All integration for experimentation
- Dataset provided in CSV + JSONL formats

---

## 📁 Project Structure

---

## 📌 File-by-File Description

### **1. convert_compound_to_intensity.py**
Converts sentiment analyzer **compound scores** into **discrete intensity levels**  
(Example: -1 to +1 → {Low, Medium, High Stress}).

### **2. dialy_dialog_trans.ipynb**
Notebook used for:
- Translating dataset dialogues  
- Data exploration  
- Manual correction and annotation  

### **3. final_dataset_complete.csv / .jsonl**
Fully processed dataset used for training + testing.

### **4. preprocess.py**
Handles all preprocessing:
- Remove stopwords  
- Cleaning URLs, emojis, punctuation  
- Tokenization  
- Lowercasing  
- Handling Telugu and code-mixed text  

### **5. final_training.py**
Model-training pipeline:
- Loads dataset  
- Vectorizes text  
- Builds classification model  
- Trains and validates  
- Saves trained model  

### **6. test_model.py**
Loads the trained model and:
- Evaluates accuracy  
- Predicts stress levels for new input  

### **7. gpt4all.py**
Uses a GPT4All model to infer stress levels conversationally.

### **8. requirements.txt**
Lists all Python dependencies.

---

## 🔧 Installation

```bash
git clone https://github.com/Dheemanth10/Telugu_Stress_bot.git
cd Telugu_Stress_bot
pip install -r requirements.txt

