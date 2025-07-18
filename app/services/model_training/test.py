from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig

# Load base model
base_model_name = "Salesforce/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)

# Load adapter
adapter_path = "../../models/fine_tuned_model"
model = PeftModel.from_pretrained(base_model, adapter_path)

test_code = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
"""

inputs = tokenizer(test_code, return_tensors="pt", max_length=512, truncation=True)

# Generate output from the model
outputs = model.generate(**inputs, max_length=128)

# Decode the generated tokens
docstring = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated Documentation:")
print(docstring)
