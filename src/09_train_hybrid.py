import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from gensim.models import Word2Vec

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed/")
MODELS_PATH = os.path.join(BASE_DIR, "../models/")
MAX_SEQUENCE_LENGTH = 50

print("Loading data and Word2Vec dictionary...")
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")).dropna(subset=['cleaned_text'])
w2v_model = Word2Vec.load(os.path.join(MODELS_PATH, "word2vec_disaster.model"))

# --- 1. DATA PREPARATION ---
print("Tokenizing and padding sequences...")
X = df['cleaned_text'].astype(str).tolist()
y = df['urgency'].tolist()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train_text, X_test_text, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_text)
vocab_size = len(tokenizer.word_index) + 1

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_SEQUENCE_LENGTH)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_SEQUENCE_LENGTH)

# --- 2. CREATE EMBEDDING MATRIX ---
print("Mapping Word2Vec vectors to Neural Network weights...")
embedding_matrix = np.zeros((vocab_size, 100)) 
for word, i in tokenizer.word_index.items():
    if word in w2v_model.wv:
        embedding_matrix[i] = w2v_model.wv[word]

# --- 3. CLASS WEIGHTS (Fixing Imbalance) ---
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# --- 4. BUILD THE HYBRID CNN-BiLSTM MODEL ---
print("\nConstructing Hybrid CNN-BiLSTM Architecture...")
model = Sequential([
    
    Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], 
              input_length=MAX_SEQUENCE_LENGTH, trainable=False),
    
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),

    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5), 
    
    Dense(32, activation='relu'),
    Dense(3, activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. TRAIN THE MODEL ---
print("\nTraining Hybrid Model (This will take a few minutes)...")
history = model.fit(X_train_pad, y_train, epochs=5, batch_size=64, 
                    validation_data=(X_test_pad, y_test), class_weight=class_weight_dict)

# --- 6. EVALUATE ---
print("\nEvaluating Hybrid Model on Test Set...")
y_pred_probs = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_probs, axis=1)

print(f"\n🚀 Hybrid Model Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\n--- Detailed Classification Report ---")
target_names = label_encoder.inverse_transform([0, 1, 2])
print(classification_report(y_test, y_pred, target_names=target_names))

model.save(os.path.join(MODELS_PATH, "hybrid_cnn_bilstm.h5"))
with open(os.path.join(MODELS_PATH, "dl_tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)
with open(os.path.join(MODELS_PATH, "dl_label_encoder.pkl"), "wb") as f:
    pickle.dump(label_encoder, f)
print("\n💾 Hybrid Model, Tokenizer, and Encoders saved successfully!")