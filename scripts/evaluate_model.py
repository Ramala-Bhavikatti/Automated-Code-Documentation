import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from sacrebleu import corpus_bleu
from rouge import Rouge
import numpy as np
from tqdm import tqdm
import os
from prompts.COTFewShot import COT_FEWSHOT_TEMPLATE

def load_test_dataset(file_path):
    """Load test dataset from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to HuggingFace dataset format
    dataset = Dataset.from_dict({
        'input_text': [item['code'] for item in data],
        'target_text': [item['docstring'] for item in data]
    })
    
    return dataset

def evaluate_model(model_path, test_dataset_path):
    """Evaluate the trained model"""
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = load_test_dataset(test_dataset_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    references = []
    
    for item in tqdm(test_dataset):
        # Prepare input
        # input_text = f"Document the following Python code:\n{item['input_text']}"
        input_text = COT_FEWSHOT_TEMPLATE.format(code=item['input_text'].strip())
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate prediction
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(prediction)
        references.append(item['target_text'])
    
    # Calculate metrics
    print("\nCalculating metrics...")
    
    # BLEU score
    bleu_score = corpus_bleu(predictions, [references])
    print(f"BLEU Score: {bleu_score.score:.4f}")
    
    # ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(predictions, references, avg=True)
    print("\nROUGE Scores:")
    for metric, scores in rouge_scores.items():
        print(f"{metric}:")
        for score_type, value in scores.items():
            print(f"  {score_type}: {value:.4f}")
    
    # Save results
    results = {
        'bleu_score': bleu_score.score,
        'rouge_scores': rouge_scores,
        'predictions': predictions,
        'references': references
    }
    
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation complete! Results saved to evaluation_results/metrics.json")

if __name__ == "__main__":
    model_path = "models/final"
    test_dataset_path = "data/docstring_dataset.json"
    
    evaluate_model(model_path, test_dataset_path) 