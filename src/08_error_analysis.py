import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

PROCESSED_DATA_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\data\\processed"
MODELS_PATH = "E:\\3rd_year\\6SEM\\Project\\Disaster_NLP_Project\\models"

# --- 1. LOAD DATA & MODEL ---
print("Loading data and model for Error Analysis...")

df_humaid = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")).dropna(subset=['cleaned_text'])

X_train_text, X_test_text, y_train, y_test = train_test_split(
    df_humaid['cleaned_text'], df_humaid['urgency'], 
    test_size=0.2, random_state=42, stratify=df_humaid['urgency']
)

with open(os.path.join(PROCESSED_DATA_PATH, "X_test_tfidf.pkl"), "rb") as f:
    X_test_tfidf = pickle.load(f)
with open(os.path.join(MODELS_PATH, "logistic_regression_model.pkl"), "rb") as f:
    model = pickle.load(f)

# --- 2. MAKE PREDICTIONS ---
print("Generating predictions...")
y_pred = model.predict(X_test_tfidf)

# --- 3. FIND THE MISTAKES ---
print("Isolating misclassifications...")
results_df = pd.DataFrame({
    'Tweet_Text': X_test_text,
    'Actual_Urgency': y_test,
    'Predicted_Urgency': y_pred
})

errors_df = results_df[results_df['Actual_Urgency'] != results_df['Predicted_Urgency']]

# Specifically isolate the most dangerous errors: Actual High, Predicted Low
critical_misses = errors_df[(errors_df['Actual_Urgency'] == 'High') & (errors_df['Predicted_Urgency'] == 'Low')]

print(f"\nTotal Mistakes Made: {len(errors_df)}")
print(f"🚨 CRITICAL MISSES (Actual High, Predicted Low): {len(critical_misses)}")

# --- 4. EXPORT TO CSV ---
errors_path = os.path.join(PROCESSED_DATA_PATH, "misclassified_tweets.csv")
critical_path = os.path.join(PROCESSED_DATA_PATH, "critical_misses.csv")

errors_df.to_csv(errors_path, index=False)
critical_misses.to_csv(critical_path, index=False)

print(f"\n✅ Error Analysis Complete!")
print(f"📂 Saved all errors to: {errors_path}")
print(f"📂 Saved critical misses to: {critical_path}")