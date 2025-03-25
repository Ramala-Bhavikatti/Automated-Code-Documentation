import os
import json
from pathlib import Path

def create_test_dataset():
    """Create a tiny test dataset for development"""
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Create a tiny test dataset with 5 examples
    test_data = [
        {
            "code": """
def calculate_sum(a, b):
    \"\"\"Calculate the sum of two numbers.
    
    Args:
        a (int): First number
        b (int): Second number
        
    Returns:
        int: Sum of a and b
    \"\"\"
    return a + b
""",
            "function_name": "calculate_sum",
            "documentation": "Calculate the sum of two numbers.",
            "language": "python"
        },
        {
            "code": """
def greet(name):
    \"\"\"Generate a greeting message.
    
    Args:
        name (str): Name of the person to greet
        
    Returns:
        str: Greeting message
    \"\"\"
    return f"Hello, {name}!"
""",
            "function_name": "greet",
            "documentation": "Generate a greeting message.",
            "language": "python"
        },
        {
            "code": """
def is_palindrome(text):
    \"\"\"Check if a string is a palindrome.
    
    Args:
        text (str): String to check
        
    Returns:
        bool: True if text is a palindrome, False otherwise
    \"\"\"
    return text == text[::-1]
""",
            "function_name": "is_palindrome",
            "documentation": "Check if a string is a palindrome.",
            "language": "python"
        },
        {
            "code": """
def count_words(text):
    \"\"\"Count the number of words in a text.
    
    Args:
        text (str): Text to count words in
        
    Returns:
        int: Number of words
    \"\"\"
    return len(text.split())
""",
            "function_name": "count_words",
            "documentation": "Count the number of words in a text.",
            "language": "python"
        },
        {
            "code": """
def reverse_string(text):
    \"\"\"Reverse a string.
    
    Args:
        text (str): String to reverse
        
    Returns:
        str: Reversed string
    \"\"\"
    return text[::-1]
""",
            "function_name": "reverse_string",
            "documentation": "Reverse a string.",
            "language": "python"
        }
    ]
    
    # Save to JSON file
    with open(data_dir / "test_dataset.json", "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nCreated test dataset with {len(test_data)} examples")
    print("This is a tiny test dataset for development. For full training, you'll need more data.")


if __name__ == "__main__":
    create_test_dataset() 