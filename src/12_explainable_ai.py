import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lime.lime_text import LimeTextExplainer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed/")
MODELS_PATH = os.path.join(BASE_DIR, "../models/")
ROBERTA_DIR = os.path.join(MODELS_PATH, "roberta_urgency_classifier")

print("🔍 Initializing XAI Diagnostic Tool (LIME)...")

# --- 1. LOAD DATA & RECREATE TEST SPLIT ---
print("Loading test data to find a misclassified tweet...")
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")).dropna(subset=['cleaned_text', 'urgency'])

X = df['cleaned_text'].astype(str).tolist()
y = df['urgency'].tolist()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
classes = label_encoder.classes_

_, X_test, _, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)

# --- 2. LOAD ROBERTA ---
print("Loading Fine-Tuned RoBERTa Brain...")
tokenizer = AutoTokenizer.from_pretrained(ROBERTA_DIR)
model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_DIR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# --- 3. LIME PREDICTION FUNCTION ---
def predictor_wrapper(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
    return probs

# --- 4. HUNTING FOR A SPECIFIC BLIND SPOT (MEMORY-SAFE) ---
print("Hunting for a tweet where RoBERTa confused 'High' and 'Medium' urgency...")
test_subset = X_test
true_subset = y_test

batch_size = 32
subset_probs = []

print("Predicting test set in batches to protect GPU VRAM...")
for i in range(0, len(test_subset), batch_size):
    batch_texts = test_subset[i:i+batch_size]
    batch_probs = predictor_wrapper(batch_texts)
    subset_probs.extend(batch_probs)

subset_probs = np.array(subset_probs)
subset_preds = np.argmax(subset_probs, axis=1)

target_idx = -1
for i in range(len(test_subset)):
    true_label = classes[true_subset[i]]
    pred_label = classes[subset_preds[i]]
    
    # We are looking for a false negative: A High urgency event ignored as Medium
    if true_label == 'High' and pred_label == 'Medium':
        target_idx = i
        break

if target_idx == -1:
    print("Could not find a High->Medium misclassification! Your model is incredibly accurate.")
    target_idx = 0

text_to_explain = test_subset[target_idx]
actual_label = classes[true_subset[target_idx]]
predicted_label = classes[subset_preds[target_idx]]

print("\n" + "="*50)
print(f"🚨 FOUND TARGET FOR DIAGNOSTICS:")
print(f"Text: '{text_to_explain}'")
print(f"Human labeled it: {actual_label}")
print(f"RoBERTa guessed: {predicted_label}")
print("="*50)

# --- 5. GENERATE THE LIME EXPLANATION ---
print("\nGenerating visual explanation... (Running 500 perturbations)")
explainer = LimeTextExplainer(class_names=classes)

# We ask LIME to explain the prediction for the specific label RoBERTa *actually* guessed
exp = explainer.explain_instance(
    text_to_explain, 
    predictor_wrapper, 
    num_features=10, 
    num_samples=500,
    top_labels=1
)

# --- 6. EXPORT VISUALIZATION ---
output_path = os.path.join(BASE_DIR, "lime_explanation.html")
exp.save_to_file(output_path)
print(f"\n✅ Diagnostic Complete! Open this file in your web browser to see the AI's brain:")
print(f"➡️ {os.path.abspath(output_path)}")