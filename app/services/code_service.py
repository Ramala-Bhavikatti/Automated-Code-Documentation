import ast
from typing import Dict, Any
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from radon.complexity import cc_visit

class CodeService:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")

    def analyze_code(self, code: str) -> Dict[str, Any]:
        try:
            tree = ast.parse(code)
            complexity = self._calculate_complexity_with_radon(code)

            return {
                'functions': self._count_functions(tree),
                'classes': self._count_classes(tree),
                "lines": self._count_effective_lines(code),
                'complexity': complexity,
                'tokens': self._tokenize_code(code),
            }
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {str(e)}")

    def _count_functions(self, tree: ast.AST) -> int:
        return len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])

    def _count_classes(self, tree: ast.AST) -> int:
        return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])

    def _count_effective_lines(self, code: str) -> int:
        lines = code.splitlines()
        meaningful_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
        return len(meaningful_lines)

    def _calculate_complexity_with_radon(self, code: str) -> int:
        """Use radon to calculate cyclomatic complexity"""
        try:
            results = cc_visit(code)
            total_complexity = sum([block.complexity for block in results])
            return total_complexity
        except Exception as e:
            print(f"Error calculating complexity with radon: {e}")
            return 0

    def _tokenize_code(self, code: str) -> list:
        doc = self.nlp(code)
        return [token.text for token in doc if not token.is_space and not token.is_punct]

    def _get_code_embeddings(self, code: str) -> list:
        inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
