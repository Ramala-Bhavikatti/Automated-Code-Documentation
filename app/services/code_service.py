import ast
from typing import Dict, Any
import spacy
from transformers import AutoTokenizer, AutoModel
import torch

class CodeService:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
    
    def analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code structure and complexity"""
        try:
            tree = ast.parse(code)
            analysis = {
                'functions': self._count_functions(tree),
                'classes': self._count_classes(tree),
                'complexity': self._calculate_complexity(tree),
                'tokens': self._tokenize_code(code),
                'embeddings': self._get_code_embeddings(code)
            }
            return analysis
        except SyntaxError as e:
            raise ValueError(f"Invalid Python code: {str(e)}")
    
    def _count_functions(self, tree: ast.AST) -> int:
        """Count number of functions in the code"""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
    
    def _count_classes(self, tree: ast.AST) -> int:
        """Count number of classes in the code"""
        return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        return complexity
    
    def _tokenize_code(self, code: str) -> list:
        """Tokenize code using spaCy"""
        doc = self.nlp(code)
        return [token.text for token in doc]
    
    def _get_code_embeddings(self, code: str) -> list:
        """Get code embeddings using CodeBERT"""
        inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().tolist() 