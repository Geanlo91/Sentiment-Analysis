import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import evaluate
import torch
from sklearn.metrics import f1_score, accuracy_score 
import numpy as np

MAX_LENGTH = 152
BATCH_SIZE = 32
EPOCHS = 5   
BASE_MODEL_NAME = "distilbert-base-uncased"
FINE_TUNED_MULTILABEL_MODEL_DIR = "fine_tuned_multilabel_model"

# ----------------------------
# Load and clean dataset
# ----------------------------
df = pd.read_csv(
    "sarcasm_data.csv",
    quotechar='"',
    escapechar='\\',
    engine="python",
    on_bad_lines="skip"
)

for col in ["sarcastic", "irony", "multipolarity"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    else:
        raise KeyError(f"Column '{col}' not found in CSV!")

df = df.dropna(subset=["sarcastic", "irony", "multipolarity"])
df[["sarcastic", "irony", "multipolarity"]] = df[["sarcastic", "irony", "multipolarity"]].astype(int)

print(f"Dataset cleaned: {len(df)} rows remaining")

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# ----------------------------
# Tokenization
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["review"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

dataset = dataset.map(tokenize, batched=True)

dataset = dataset.map(
    lambda x: {
        "labels": [
            float(x["sarcastic"]),
            float(x["irony"]),
            float(x["multipolarity"])
        ]
    }
)

dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# ----------------------------
# Custom thresholds per label
# ----------------------------

LABEL_THRESHOLDS = {
    "sarcastic": 0.4,     # lower threshold for sarcasm
    "irony": 0.5,         # keep default
    "multipolarity": 0.35 # more sensitive for multipolarity
}

label_order = ["sarcastic", "irony", "multipolarity"]
thresholds = np.array([LABEL_THRESHOLDS[l] for l in label_order])

# ----------------------------
# Metrics
# ----------------------------
f1 = evaluate.load("f1")
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()

    # apply per-label thresholds
    preds = (probs > thresholds).astype(int)

    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)

    return {
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "accuracy": acc
    }

# ----------------------------
# Model
# ----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME, 
    num_labels=3,
    problem_type="multi_label_classification"
)

# ----------------------------
# Training Arguments
# ----------------------------
training_args = TrainingArguments(
    output_dir="multilabel_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,  
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS, 
    weight_decay=0.01,
    warmup_steps=50,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",  
    greater_is_better=True,
    save_total_limit=2,
)

# ----------------------------
# Trainer
# ----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ----------------------------
# Train & Save
# ----------------------------
trainer.train()
trainer.save_model(FINE_TUNED_MULTILABEL_MODEL_DIR)
tokenizer.save_pretrained(FINE_TUNED_MULTILABEL_MODEL_DIR)
print(f"Multi-label model saved to {FINE_TUNED_MULTILABEL_MODEL_DIR}")

