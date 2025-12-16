# train_legal.py
# Full training pipeline for Kenyan Legal AI Model
# Works with your KenyaLaw Scraper v6.0 output
# --- MODIFIED FOR CPU-ONLY TRAINING (NO UNSLOTH, NO GPU) ---

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset

# --- Standard Transformers/PEFT Imports ---
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model
)
import torch
import gc

# Check if CUDA is available and warn if not (as it's CPU-only)
if torch.cuda.is_available():
    print("WARNING: GPU is available, but this script is configured for CPU-only.")
    print("To use GPU, please revert changes related to 'bnb_config', 'fp16', etc.")
else:
    print("GPU not detected. Running on CPU. This will be very slow.")

# ========================= CONFIG =========================
class TrainConfig:
    BASE_DIR = Path(os.path.expanduser("~/projects/kenya_law/data"))
    OUTPUT_DIR = Path("./kenya-legal-llm")
    
    # --- MODIFIED FOR CPU: Switched to a much smaller model ---
    # Training an 8B model on CPU is not feasible. 1.1B is a testable alternative.
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # --- MODIFIED FOR CPU: Reduced max_seq_length to save RAM ---
    MAX_SEQ_LENGTH = 2048 # 8192 is too large for most CPU/RAM setups
    BATCH_SIZE = 1 # Keep batch size low for CPU
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    WARMUP_STEPS = 10
    LOGGING_STEPS = 10
    SAVE_STEPS = 100
    EVAL_STRATEGY = "no"

    # Output formats
    HF_OUTPUT = OUTPUT_DIR / "hf_adapter" # Saving adapter here

cfg = TrainConfig()
cfg.OUTPUT_DIR.mkdir(exist_ok=True)
cfg.HF_OUTPUT.mkdir(exist_ok=True)

# ========================= LOAD DATA =========================
# (Data loading functions remain unchanged)
def load_jsonl(file_path: Path) -> List[Dict]:
    data = []
    if not file_path.exists():
        print(f"Warning: {file_path} not found!")
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    continue
    print(f"Loaded {len(data)} cases from {file_path.name}")
    return data

def load_json(file_path: Path) -> Dict:
    if not file_path.exists():
        print(f"Warning: {file_path} not found!")
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

print("Loading Kenyan legal data...")
cases = load_jsonl(cfg.BASE_DIR / "kenya_law_training_data.jsonl")
constitution = load_json(cfg.BASE_DIR / "constitution.json")
acts = load_json(cfg.BASE_DIR / "acts_of_kenya.json")
subsidiary = load_json(cfg.BASE_DIR / "subsidiary_legislation.json")
counties = load_json(cfg.BASE_DIR / "county_legislation.json")

# ========================= PREPARE DATA =========================
# (Data prep functions remain unchanged)
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def create_instruction_samples() -> List[Dict[str, Any]]:
    samples = []

    # 1. Constitution
    for title, content in constitution.items():
        chunks = chunk_text(content, 1200)
        for i, chunk in enumerate(chunks):
            samples.append({
                "instruction": f"You are a Kenyan constitutional law expert. Answer based on the Constitution of Kenya 2010.",
                "input": f"Explain: {title}" + (f" (Part {i+1})" if len(chunks) > 1 else ""),
                "output": chunk[:4000]
            })

    # 2. Acts of Parliament
    for title, content in acts.items():
        if len(content.split()) > 200:
            samples.append({
                "instruction": "You are a Kenyan lawyer. Cite and explain the relevant law.",
                "input": f"What does the law say about: {title}?",
                "output": content[:6000]
            })

    # 3. Case Law (High-value)
    for case in cases[:2000]:  # Use top 2000 cases
        text = case.get("text", "")
        if len(text.split()) < 200:
            continue
        metadata = case.get("metadata", {})
        court = metadata.get("court", "Kenyan Court")
        date = metadata.get("date", "Unknown date")

        samples.append({
            "instruction": "You are a Kenyan judge. Analyze this case and give legal reasoning.",
            "input": f"Case: {case['case_name']}\nCourt: {court}\nDate: {date}\n\n{text[:3000]}...",
            "output": f"**Case Analysis:**\n\n**Citation:** {case['case_name']}\n**Court:** {court}\n**Date:** {date}\n\n**Legal Reasoning:**\n{text[1000:8000]}\n\n**Held:** {text.split('Held:')[-1].split('JUDGMENT')[0] if 'Held:' in text else 'See full judgment.'}"
        })

    # 4. County Laws (PDF + HTML)
    for county, data in counties.items():
        for law_name, law in data.get("laws", {}).items():
            content = law.get("content", "")
            if "UNABLE_TO_EXTRACT" in content or len(content.split()) < 100:
                continue
            samples.append({
                "instruction": f"You are a legal expert in {county} County legislation.",
                "input": f"What does {county} law say about: {law_name}?",
                "output": content[:5000]
            })

    print(f"Created {len(samples)} training samples")
    return samples

# ========================= FORMAT FOR TRAINING =========================
# (Formatting function remains unchanged)
def format_alpaca(sample: Dict) -> str:
    return f"""### Instruction:
{sample['instruction']}

### Input:
{sample['input']}

### Response:
{sample['output']}"""

print("Creating dataset...")
raw_samples = create_instruction_samples()
formatted = [format_alpaca(s) for s in raw_samples]
dataset = Dataset.from_dict({"text": formatted})

# ========================= LOAD MODEL =========================
print(f"Loading {cfg.MODEL_NAME} for CPU...")

# --- MODIFIED FOR CPU: Removed all BitsAndBytesConfig and quantization ---
# Loading the model in full precision (float32) on the CPU.
# This will consume a lot of RAM.
model = AutoModelForCausalLM.from_pretrained(
    cfg.MODEL_NAME,
    trust_remote_code=True,
    # No quantization, no device_map
)

# --- MODIFIED: Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- MODIFIED: Setup PEFT (LoRA) ---
print("Setting up PEFT (LoRA)...")

# --- MODIFIED FOR CPU: Removed 'prepare_model_for_kbit_training' ---
# model = prepare_model_for_kbit_training(model) # This is for k-bit (GPU)

# Define LoRA configuration
# Target modules are for TinyLlama.
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply PEFT to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ========================= TRAINER =========================
print("Setting up SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=cfg.MAX_SEQ_LENGTH,
    dataset_num_proc=2,
    packing=True, # Pack multiple short sequences into one
    args=TrainingArguments(
        per_device_train_batch_size=cfg.BATCH_SIZE,
        gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=cfg.WARMUP_STEPS,
        num_train_epochs=cfg.NUM_EPOCHS,
        learning_rate=cfg.LEARNING_RATE,
        
        # --- MODIFIED FOR CPU ---
        fp16=False,
        bf16=False,
        optim="adamw_torch", # Standard CPU-compatible optimizer
        # ------------------------

        logging_steps=cfg.LOGGING_STEPS,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=str(cfg.OUTPUT_DIR),
        report_to="none",
        save_strategy="steps",
        save_steps=cfg.SAVE_STEPS,
        eval_strategy=cfg.EVAL_STRATEGY,
        gradient_checkpointing=True, # Still useful to save RAM
        load_best_model_at_end=False,
        save_total_limit=3,
    ),
)

print("Starting training... (This will be very slow on CPU)")
trainer.train()

# ========================= SAVE MODEL =========================
print(f"Saving PEFT adapter model to {cfg.HF_OUTPUT}...")
trainer.save_model(str(cfg.HF_OUTPUT))
tokenizer.save_pretrained(str(cfg.HF_OUTPUT))
print("Adapter model saved.")

# ========================= EXPORT TO GGUF (MANUAL STEP) =========================
print("\n--- GGUF CONVERSION (MANUAL STEP) ---")
print(f"Training complete. PEFT adapter saved to: {cfg.HF_OUTPUT}")
print("GGUF conversion is NOT performed by this script.")
print("\nTo create a GGUF (for Ollama):")
print("1. Merge the adapter weights with the base model and save it in HF format.")
print("   (You can write a separate python script using model.merge_and_unload())")
print("2. Use a tool like 'llama.cpp' to quantize and convert the merged HF model to GGUF.")
print("--- END OF SCRIPT ---")