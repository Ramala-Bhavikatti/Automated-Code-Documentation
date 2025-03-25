import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from pathlib import Path
import os
from tqdm import tqdm

def load_dataset(file_path):
    """Load dataset from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to HuggingFace dataset format
    dataset = Dataset.from_dict({
        'input_text': [item['code'] for item in data],
        'target_text': [item['docstring'] for item in data]
    })
    
    return dataset

def preprocess_function(examples, tokenizer):
    """Preprocess examples for training"""
    inputs = [f"Document the following Python code:\n{code}" for code in examples["input_text"]]
    targets = examples["target_text"]
    
    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        targets,
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_model():
    """Train the code documentation model"""
    # Load model and tokenizer
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset("data/codesearchnet_train.json")
    eval_dataset = load_dataset("data/docstring_dataset.json")
    
    # Preprocess datasets
    print("Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/trained",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=100,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        push_to_hub=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save model
    print("Saving model...")
    trainer.save_model("models/final")
    tokenizer.save_pretrained("models/final")
    
    print("Training complete!")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models/trained", exist_ok=True)
    os.makedirs("models/final", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Train model
    train_model() 