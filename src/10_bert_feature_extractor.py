import os
import json
import random

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "../data/processed")
MODELS_PATH = os.path.join(BASE_DIR, "../models")
INPUT_FILE = os.path.join(PROCESSED_DATA_PATH, "humaid_cleaned.csv")

MODEL_NAME = os.getenv("TRANSFORMER_MODEL", "roberta-base")
MAX_LENGTH = int(os.getenv("MAX_TOKEN_LENGTH", "128"))
EPOCHS = int(os.getenv("TRAIN_EPOCHS", "5"))
SEED = int(os.getenv("TRAIN_SEED", "456"))
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "0"))
USE_CLASS_LABEL_MAPPING = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
        )
        logits = outputs.get("logits")
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=MAX_LENGTH)


def main():
    set_seed(SEED)
    os.makedirs(MODELS_PATH, exist_ok=True)

    print("Loading cleaned dataset...")
    raw_df = pd.read_csv(INPUT_FILE).dropna(subset=["cleaned_text", "urgency"])

    if USE_CLASS_LABEL_MAPPING and "class_label" in raw_df.columns:
        class_to_urgency = (
            raw_df[["class_label", "urgency"]]
            .dropna()
            .drop_duplicates()
            .groupby("class_label")["urgency"]
            .agg(lambda values: values.mode().iloc[0])
            .to_dict()
        )

        mapped = raw_df["class_label"].map(class_to_urgency)
        mapping_acc = accuracy_score(raw_df["urgency"], mapped)

        print("\nRule-based class_label -> urgency mapping accuracy:")
        print(f"Overall Accuracy: {mapping_acc:.2%}")
        print("Detailed Classification Report:")
        print(classification_report(raw_df["urgency"], mapped, zero_division=0))

        mapping_out = os.path.join(MODELS_PATH, "classlabel_to_urgency_mapping.json")
        with open(mapping_out, "w", encoding="utf-8") as f:
            json.dump(class_to_urgency, f, indent=2)

        print(f"\nSaved deterministic mapping to: {mapping_out}")
        print("Set USE_CLASS_LABEL_MAPPING=0 to force text-only transformer training.")
        return

    df = raw_df[["cleaned_text", "urgency"]].rename(columns={"cleaned_text": "text", "urgency": "label"})

    if SAMPLE_SIZE > 0 and SAMPLE_SIZE < len(df):
        df = df.sample(n=SAMPLE_SIZE, random_state=SEED)
        print(f"Using sampled rows: {len(df)}")

    classes = sorted(df["label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(classes)}
    id2label = {idx: label for label, idx in label2id.items()}
    df["label"] = df["label"].map(label2id)

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=SEED, stratify=df["label"])
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.15,
        random_state=SEED,
        stratify=train_df["label"],
    )

    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}")
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(classes),
        id2label=id2label,
        label2id=label2id,
    )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    columns = ["input_ids", "attention_mask", "label"]
    train_ds.set_format(type="torch", columns=columns)
    val_ds.set_format(type="torch", columns=columns)
    test_ds.set_format(type="torch", columns=columns)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(sorted(train_df["label"].unique())),
        y=train_df["label"].values,
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_PATH, "roberta_urgency_checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16 if use_cuda else 8,
        per_device_eval_batch_size=32 if use_cuda else 16,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=use_cuda,
        logging_steps=100,
        report_to="none",
        seed=SEED,
    )

    trainer = WeightedLossTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nTraining RoBERTa classifier...")
    trainer.train()

    print("\nEvaluating best checkpoint on holdout test set...")
    preds = trainer.predict(test_ds)
    y_true = test_df["label"].to_numpy()
    y_pred = np.argmax(preds.predictions, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Test Accuracy: {acc:.2%}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))

    final_model_dir = os.path.join(MODELS_PATH, "roberta_urgency_seed456")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    with open(os.path.join(final_model_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

    print(f"\nSaved model and tokenizer to: {final_model_dir}")


if __name__ == "__main__":
    main()