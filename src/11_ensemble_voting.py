import os
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed")
MODELS_PATH = os.path.join(BASE_DIR, "../models")

ROBERTA_DIR = os.path.join(MODELS_PATH, "roberta_urgency_classifier")
CNN_MODEL_PATH = os.path.join(MODELS_PATH, "hybrid_cnn_bilstm.h5")
CNN_TOKENIZER_PATH = os.path.join(MODELS_PATH, "dl_tokenizer.pkl")
LOGREG_MODEL_PATH = os.path.join(MODELS_PATH, "logistic_regression_model.pkl")
TFIDF_PATH = os.path.join(MODELS_PATH, "tfidf_vectorizer.pkl")  

print("🚀 Initializing the Ensemble 'Meta-Model'...")

# --- 1. LOAD TEST DATA ---
print("Loading the test dataset...")
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")).dropna(subset=['cleaned_text', 'urgency'])

X = df['cleaned_text'].astype(str).tolist()
y = df['urgency'].tolist()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)

ensemble_probs = np.zeros((len(X_test), 3))
active_models = 0

# --- 2. THE ROBERTA VOTE (Weight: 60%) ---
if os.path.exists(ROBERTA_DIR):
    print("\n🧠 Loading RoBERTa (Weight: 60%)...")
    tokenizer_rob = AutoTokenizer.from_pretrained(ROBERTA_DIR)
    model_rob = AutoModelForSequenceClassification.from_pretrained(ROBERTA_DIR)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_rob.to(device)
    model_rob.eval()

    rob_probs = []
    batch_size = 32
    print("Gathering RoBERTa's predictions...")
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_texts = X_test[i:i+batch_size]
            inputs = tokenizer_rob(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
            outputs = model_rob(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
            rob_probs.extend(probs)
            
    ensemble_probs += np.array(rob_probs) * 0.60
    active_models += 1
else:
    print("\n⚠️ RoBERTa model not found. Skipping...")

# --- 3. THE CNN-BiLSTM VOTE (Weight: 25%) ---
if os.path.exists(CNN_MODEL_PATH) and os.path.exists(CNN_TOKENIZER_PATH):
    print("\n🧠 Loading CNN-BiLSTM (Weight: 25%)...")
    model_cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
    with open(CNN_TOKENIZER_PATH, 'rb') as f:
        tokenizer_cnn = pickle.load(f)
    
    print("Gathering CNN-BiLSTM's predictions...")
    X_test_seq = tokenizer_cnn.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=50) 
    
    cnn_probs = model_cnn.predict(X_test_pad, verbose=0)
    ensemble_probs += cnn_probs * 0.25
    active_models += 1
else:
    print("\n⚠️ CNN-BiLSTM model not found. Skipping...")

# --- 4. THE LOGISTIC REGRESSION VOTE (Weight: 15%) ---
if os.path.exists(LOGREG_MODEL_PATH) and os.path.exists(TFIDF_PATH):
    print("\n🧠 Loading Logistic Regression (Weight: 15%)...")
    with open(LOGREG_MODEL_PATH, 'rb') as f:
        model_logreg = pickle.load(f)
    with open(TFIDF_PATH, 'rb') as f:
        tfidf = pickle.load(f)
        
    print("Gathering Logistic Regression's predictions...")
    X_test_tfidf = tfidf.transform(X_test)
    logreg_probs = model_logreg.predict_proba(X_test_tfidf)
    
    ensemble_probs += logreg_probs * 0.15
    active_models += 1
else:
    print("\n⚠️ Logistic Regression / TF-IDF files not found. Skipping...")

# --- 5. THE FINAL ENSEMBLE DECISION ---
if active_models == 0:
    print("\n❌ Error: No models were found to create the ensemble.")
    exit()

print(f"\n✅ Aggregated votes from {active_models} models.")

ensemble_probs = ensemble_probs / np.sum(ensemble_probs, axis=1, keepdims=True)

y_pred_ensemble = np.argmax(ensemble_probs, axis=1)

print("\n" + "="*50)
print(f"🏆 ENSEMBLE MODEL ACCURACY: {accuracy_score(y_test, y_pred_ensemble):.2%}")
print("="*50)

print("\n--- Detailed Classification Report ---")

classes = sorted(list(set(y)))
print(classification_report(y_test, y_pred_ensemble, target_names=classes))