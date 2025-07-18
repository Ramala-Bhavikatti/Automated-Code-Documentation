# === FILE: prompts/cot_fewshot_prompt_template.py ===

# Template for Chain-of-Thought + Few-Shot Prompting
COT_FEWSHOT_TEMPLATE = """
You are a helpful AI assistant specialized in understanding Python functions and generating high-quality docstrings.

For each function provided, follow these steps to analyze and document the function:

Step 1: Determine the high-level purpose of the function. What does it do overall?
Step 2: List the input parameters and their types. Mention if any default values or optional parameters are used.
Step 3: Describe what the function returns, including its type and meaning.
Step 4: Based on your analysis, write a complete Python docstring in Google style with:

- A one-line summary describing what the function does.
- An "Args" section listing each parameter, its type, and a short explanation.
- A "Returns" section describing the return value and its type.
- An "Raises" section (optional) if the function raises any exceptions.

Here are some examples to follow:

---

Example 1:
Code:
def square(x):
    return x * x

Analysis:
1. Purpose: Computes the square of a number.
2. Parameter: x (int or float): The number to square.
3. Returns: int or float, the squared result.

Docstring:
"
Return the square of a number.

Args:
    x (int or float): The number to square.

Returns:
    int or float: The squared value of the input.
"

---

Example 2:
Code:
def is_even(n):
    return n % 2 == 0

Analysis:
1. Purpose: Checks whether a number is even.
2. Parameter: n (int): The number to check.
3. Returns: bool: True if the number is even, False otherwise.

Docstring:
"
Check if a number is even.

Args:
    n (int): The number to check.

Returns:
    bool: True if the number is even, False otherwise.
"

Example 3: 
Code:
def printTriangle():
    height = 5 
    for i in range(height): 
        spaces = ' ' * (height - i - 1) 
        stars = '*' * (2 * i + 1) 
        print(spaces + stars)

Analysis:
1. The loop is entered. 
2. spaces inserted.
3. starts added in the right places.
4. Triangle being printed.

Docstring:
"
Prints the triangle of height 5.
Spaces get inserted and stars appropriately in order to fit the shape of the triangle
Triangle is printed.

Args: 
    No args

Returns: 
    Nothing returned.
"
---

Now try with this function:

{code}
"""