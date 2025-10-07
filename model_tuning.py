import pandas as pd
import numpy as np
import re
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, EarlyStoppingCallback, Trainer, TrainingArguments
import evaluate
import os
import torch

print(torch.cuda.is_available())

MAX_LENGTH = 152
NUM_LABELS = 3
EPOCHS = 3
BATCH_SIZE = 32
BASE_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
FINE_TUNED_MODEL_DIR = "fine_tuned_sentiment_model"

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    
    # Remove URLs, email addresses, whitespace
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def validate_and_clean_data(df):
    """Comprehensive data validation and cleaning"""
    print(f"Initial dataset size: {len(df)} rows")
    
    # Check required columns
    required_cols = ["review", "label"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in CSV!")
    
    # Handle missing review text
    df = df.dropna(subset=['review'])
    
    # Clean text data
    df['review'] = df['review'].apply(clean_text)
    
    # Remove empty texts after cleaning
    df = df[df['review'].str.len() > 0]
    
    # Remove very short texts (less than 3 characters)
    df = df[df['review'].str.len() >= 3]
  
    # Remove very long texts (keep within reasonable limits)
    df = df[df['review'].str.len() <= 1000]
    
    # Convert labels to numeric, invalid ones become NaN
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    
    # Remove rows with invalid labels
    df = df.dropna(subset=['label'])
    
    # Ensure labels are within valid range (0, 1, 2 for sentiment)
    valid_labels = df['label'].isin([0, 1, 2])
    invalid_count = (~valid_labels).sum()
    if invalid_count > 0:
        print(f"Removing {invalid_count} rows with invalid label values")
        df = df[valid_labels]
    
    # Convert to int after validation
    df['label'] = df['label'].astype(int)
    
    # Remove duplicates based on review text
    initial_size = len(df)
    df = df.drop_duplicates(subset=['review'], keep='first')
    duplicates_removed = initial_size - len(df)
    print(f"Removed {duplicates_removed} duplicate reviews")
    
    # Show label distribution
    print(f"\nLabel distribution:")
    print(df['label'].value_counts().sort_index())
    
    print(f"\nFinal cleaned dataset size: {len(df)} rows")
    
    # Reset index after all the filtering
    df = df.reset_index(drop=True)
    
    return df

def tokenize(batch):
    texts = [str(x) if x is not None else "" for x in batch["review"]]
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

if __name__ == "__main__":
    # Load and clean data with robust error handling
    try:
        df = pd.read_csv(
            "training_data.csv",
            quotechar='"',
            escapechar='\\',
            engine="python",
            on_bad_lines="skip"
        )
    except Exception as e:
        print(f"Error reading CSV: {e}")
        print("Trying with basic pandas read_csv...")
        df = pd.read_csv("training_data.csv")
    
    # Apply comprehensive data cleaning
    df = validate_and_clean_data(df)
    
    # Check if we have enough data after cleaning
    if len(df) < 10:
        raise ValueError(f"Insufficient data after cleaning: only {len(df)} rows remaining")
    
    dataset = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # Use fewer processes on Windows to avoid hang
    dataset = dataset.map(tokenize, batched=True, num_proc=1)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(pred):
        logits = pred.predictions
        labels = pred.label_ids
        preds = np.argmax(logits, axis=-1)
        acc = accuracy.compute(predictions=preds, references=labels)
        f1_score = f1.compute(predictions=preds, references=labels, average="weighted")
        return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL_NAME, num_labels=NUM_LABELS)

    training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",   
    save_strategy="epoch",        
    logging_strategy="epoch",      
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    warmup_steps=50,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2
)

    model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    trainer.save_model(FINE_TUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)
    print(f"Fine-tuned model saved to {FINE_TUNED_MODEL_DIR}")
