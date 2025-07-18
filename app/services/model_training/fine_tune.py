import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

def fine_tune_model():
    # Load the pre-trained CodeT5 model and tokenizer
    model_name = "Salesforce/codet5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load CodeSearchNet dataset (Python code)
    dataset = load_dataset("code_search_net", "python")

    # Check the dataset structure (for debugging purposes)
    print(dataset)
    print(dataset['train'].column_names)

    # Tokenization function - Update to the correct field name ('func_code_string')
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["func_code_string"],  # Update this to the correct column name
            padding="max_length", 
            truncation=True, 
            max_length=512
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["func_documentation_string"],  # Update this to the correct column name
                padding="max_length", 
                truncation=True, 
                max_length=128
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply the tokenization to the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],  # You can also include "k", "o", "wi", "wo" optionally
        lora_dropout=0.1,
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Define TrainingArguments
    training_args = TrainingArguments(
        output_dir="./results", 
        learning_rate=5e-5, 
        per_device_train_batch_size=8, 
        per_device_eval_batch_size=8, 
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=500,
        save_steps=500,
        save_total_limit=3
    )

    # Trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("app/models/fine_tuned_model")
    tokenizer.save_pretrained("app/models/fine_tuned_model")

if __name__ == "__main__":
    fine_tune_model()
