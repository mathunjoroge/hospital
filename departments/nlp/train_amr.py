import os
import csv
import logging
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import torch
import numpy as np

# Explicitly avoid TensorFlow
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelTrainer")

# Import transformers with fallback
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
    )
except ImportError as e:
    if "transformers.trainer" in str(e) or "modeling_tf_utils" in str(e):
        logger.warning("TensorFlow-related import failed. Using PyTorch-only imports.")
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
        )
        # Manually import Trainer to avoid TensorFlow dependencies
        from transformers.trainer import Trainer
    else:
        raise

from datasets import Dataset
from collections import Counter

# Log environment details
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

# 1. Define model and label mapping
model_name = "emilyalsentzer/Bio_ClinicalBERT"
categories = ["amr_high", "amr_low", "amr_none", "ipc_adequate", "ipc_inadequate", "ipc_none"]
label_map = {name: idx for idx, name in enumerate(categories)}
id2label = {idx: name for idx, name in enumerate(categories)}

# 2. Load training texts from CSV
def load_training_data() -> List[str]:
    data_path = "/home/mathu/projects/hospital/departments/nlp/resources/amr_training_data.csv"
    texts = []
    try:
        with open(data_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            texts = [row[0] for row in reader if row]
        logger.info(f"Loaded {len(texts)} training texts from {data_path}")
    except FileNotFoundError:
        logger.error(f"CSV file not found: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading CSV file {data_path}: {e}")
        raise
    if not texts:
        logger.error("No texts loaded from CSV file")
        raise ValueError("No texts loaded from CSV file")
    return texts

texts = load_training_data()

# 3. Enhanced auto-labeling function with AMR/IPC keywords
def auto_label(texts: List[str]) -> List[int]:
    labels = []
    for text in texts:
        text_lower = text.lower()
        assigned = False
        keywords = {
            "amr_high": ["mrsa", "vre", "esbl", "carbapenem-resistant", "multidrug-resistant", 
                         "antibiotic failure", "antibiotic resistance", "recent antibiotic use", 
                         "treatment failure", "possible resistance", "ciprofloxacin", "ceftriaxone"],
            "amr_low": ["susceptible", "sensitive", "culture negative", "pan-sensitive"],
            "amr_none": [],  # Fallback for AMR
            "ipc_adequate": ["contact precautions", "hand hygiene", "isolation protocol", "sterile technique"],
            "ipc_inadequate": ["no isolation", "poor compliance", "no precautions", "no ipc", "no sterile technique"],
            "ipc_none": ["no signs of infection", "no antibiotics", "routine care", "no infection"]
        }
        # Prioritize AMR labels
        for category in ["amr_high", "amr_low", "amr_none"]:
            if category != "amr_none" and any(keyword in text_lower for keyword in keywords[category]):
                labels.append(label_map[category])
                assigned = True
                break
        if not assigned and any(keyword in text_lower for keyword in ["infection", "culture", "antibiotic"]):
            labels.append(label_map["amr_none"])
            assigned = True
        # Then check IPC labels
        if not assigned:
            for category in ["ipc_adequate", "ipc_inadequate", "ipc_none"]:
                if any(keyword in text_lower for keyword in keywords[category]):
                    labels.append(label_map[category])
                    assigned = True
                    break
        if not assigned:
            labels.append(label_map["ipc_none"])
            logger.debug(f"Unlabeled text: {text[:50]}...")
    return labels

labels = auto_label(texts)

# 4. Balance class distribution
label_counts = Counter(labels)
logger.info("Initial class distribution:")
for idx, count in label_counts.items():
    logger.info(f"{id2label[idx]}: {count} examples")

# Merge rare classes (<2 examples) into fallback classes
min_examples = 2
new_texts = []
new_labels = []
for text, label in zip(texts, labels):
    if label_counts[label] < min_examples:
        if label in [label_map["amr_high"], label_map["amr_low"], label_map["amr_none"]]:
            new_labels.append(label_map["amr_none"])
        else:
            new_labels.append(label_map["ipc_none"])
        logger.debug(f"Reassigned text to fallback class: {text[:50]}...")
    else:
        new_labels.append(label)
    new_texts.append(text)

texts, labels = new_texts, new_labels

# 5. Filter out examples with ambiguous labels
filtered = [(t, l) for t, l in zip(texts, labels) if l >= 0]
if not filtered:
    raise ValueError("No valid labeled examples found in dataset")
texts, labels = map(list, zip(*filtered))

# 6. Check final class distribution
label_counts = Counter(labels)
logger.info("Final class distribution:")
for idx, count in label_counts.items():
    logger.info(f"{id2label[idx]}: {count} examples")

# Ensure at least two examples per class
for idx in range(len(categories)):
    if label_counts.get(idx, 0) < min_examples:
        logger.warning(f"Class {id2label[idx]} has {label_counts.get(idx, 0)} examples. Adding synthetic examples.")
        synthetic_texts = [
            f"Synthetic {id2label[idx]} example: {'resistant infection' if 'amr' in id2label[idx] else 'no precautions' if 'ipc' in id2label[idx] else 'routine care'}."
        ] * (min_examples - label_counts.get(idx, 0))
        texts.extend(synthetic_texts)
        labels.extend([idx] * len(synthetic_texts))
        logger.info(f"Added {len(synthetic_texts)} synthetic examples for {id2label[idx]}")

# 7. Ensure class weights cover all classes
class_weights = torch.ones(len(categories)).to("cuda" if torch.cuda.is_available() else "cpu")
for idx, category in enumerate(categories):
    count = label_counts.get(idx, 0)
    if count > 0:
        class_weights[idx] = 1.0 / count
    else:
        class_weights[idx] = 1.0
logger.info(f"Class weights: {class_weights}")

# 8. Split into train and validation sets
try:
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
except ValueError as e:
    logger.warning(f"Stratified split failed: {e}. Falling back to non-stratified split.")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

# 9. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(categories), id2label=id2label, label2id=label_map
)

# 10. Tokenization function
MAX_LENGTH = 128
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

# 11. Prepare Hugging Face Datasets
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# 12. Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    per_class = precision_recall_fscore_support(labels, predictions, zero_division=0)
    per_class_metrics = {
        f"precision_{id2label[i]}": p for i, p in enumerate(per_class[0])
    }
    per_class_metrics.update({
        f"recall_{id2label[i]}": r for i, r in enumerate(per_class[1])
    })
    per_class_metrics.update({
        f"f1_{id2label[i]}": f for i, f in enumerate(per_class[2])
    })
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **per_class_metrics
    }

# 13. Training arguments
training_args = TrainingArguments(
    output_dir="/home/mathu/projects/hospital/amr_ipc_classifier",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=8,
    logging_dir="/home/mathu/projects/hospital/logs",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=2e-5,
    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_steps=10,
    save_total_limit=2
)

# 14. Trainer setup with class weights
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics
)

# 15. Train the model
logger.info("Starting training...")
trainer.train()

# 16. Save the trained model and tokenizer
model.save_pretrained("/home/mathu/projects/hospital/amr_ipc_classifier")
tokenizer.save_pretrained("/home/mathu/projects/hospital/amr_ipc_classifier")

logger.info("✅ Training complete. Model saved to /home/mathu/projects/hospital/amr_ipc_classifier")