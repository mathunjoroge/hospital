import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"  # Explicitly enable PyTorch
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset
import logging
from collections import Counter

# Set environment variables
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelTrainer")

# 1. Define model and label mapping
model_name = "emilyalsentzer/Bio_ClinicalBERT"
categories = ["amr_high", "amr_low", "amr_none", "ipc_adequate", "ipc_inadequate", "ipc_none"]
label_map = {name: idx for idx, name in enumerate(categories)}
id2label = {idx: name for idx, name in enumerate(categories)}

# 2. Load training texts
from resources.amr_training_data import texts  # Import from saved file

# 3. Enhanced auto-labeling function with AMR/IPC keywords
def auto_label(texts: List[str]) -> List[int]:
    labels = []
    for text in texts:
        text_lower = text.lower()
        assigned = False
        keywords = {
            "amr_high": ["mrsa", "vre", "esbl", "carbapenem-resistant", "multidrug-resistant", 
                        "antibiotic failure", "antibiotic resistance", "recent antibiotic use", 
                        "treatment failure", "possible resistance"],
            "amr_low": ["susceptible", "sensitive", "culture negative", "pan-sensitive"],
            "amr_none": [],  # Fallback for AMR if no specific keywords
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

# 4. Filter out examples with ambiguous labels
filtered = [(t, l) for t, l in zip(texts, labels) if l >= 0]
if not filtered:
    raise ValueError("No valid labeled examples found in dataset")
texts, labels = zip(*filtered)

# 5. Check class distribution
label_counts = Counter(labels)
logger.info("Class distribution:")
for idx, count in label_counts.items():
    logger.info(f"{id2label[idx]}: {count} examples")

# 6. Ensure class weights cover all classes
# Initialize weights for all 6 classes, defaulting to 1.0 for missing classes
class_weights = torch.ones(len(categories)).to("cuda" if torch.cuda.is_available() else "cpu")
for idx, category in enumerate(categories):
    count = label_counts.get(idx, 0)
    if count > 0:
        class_weights[idx] = 1.0 / count
    else:
        class_weights[idx] = 1.0  # Default weight for classes with zero counts
logger.info(f"Class weights: {class_weights}")

# 7. Split into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 8. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(categories), id2label=id2label, label2id=label_map
)

# 9. Tokenization function
MAX_LENGTH = 128
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

# 10. Prepare Hugging Face Datasets
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# 11. Define compute_metrics function for evaluation
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

# 12. Training arguments
training_args = TrainingArguments(
    output_dir="/home/mathu/projects/hospital/amr_ipc_classifier",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=6,
    logging_dir="./logs",
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

# 13. Trainer setup with class weights
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

# 14. Train the model
logger.info("Starting training...")
trainer.train()

# 15. Save the trained model and tokenizer
model.save_pretrained("/home/mathu/projects/hospital/amr_ipc_classifier")
tokenizer.save_pretrained("/home/mathu/projects/hospital/amr_ipc_classifier")

logger.info("✅ Training complete. Model saved to /home/mathu/projects/hospital/amr_ipc_classifier")