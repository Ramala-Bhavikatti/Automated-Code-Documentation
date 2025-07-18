import ast
from typing import Dict, Any, List, Optional
import numpy as np
import faiss
import spacy
from transformers import AutoTokenizer, AutoModel
import torch
from app.services.code_service import CodeService
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.prompts.CoTFewShot import COT_FEWSHOT_TEMPLATE
class DocumentationService:
    def __init__(self):
        self.code_service = CodeService()
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        self.embeddings_dim = 768
        self.index = None
        self.documentation_store = {}
        self.doc_model = AutoModelForSeq2SeqLM.from_pretrained("app/models/fourth_fine_tuned_model")
        self.doc_tokenizer = AutoTokenizer.from_pretrained("app/models/fourth_fine_tuned_model")

    def generate_documentation(self, code: str) -> Dict[str, Any]:
        analysis = self.code_service.analyze_code(code)
        inline_doc = self._generate_inline_documentation(code)
        semantic_doc = self._generate_semantic_documentation(code, analysis)
        prompt_doc = self._generate_prompt_based_documentation(code)

        documentation = f"""# Code Documentation

## Inline Documentation
{inline_doc}

## Semantic Analysis
{semantic_doc}

## Prompt-Based Documentation
{prompt_doc}
"""
        self._store_documentation(code, documentation)

        return {
            'documentation': documentation,
            'analysis': analysis
        }

    def _generate_inline_documentation(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            lines = code.split('\n')
            documented_lines = []

            for i, line in enumerate(lines):
                node = self._find_node_at_line(tree, i + 1)

                if isinstance(node, ast.ClassDef):
                    documented_lines.append(f'# Class: {node.name}')
                    documented_lines.append(line)

                elif isinstance(node, ast.FunctionDef):
                    documented_lines.append(f'# Function: {node.name}')
                    documented_lines.append(line)

                elif isinstance(node, (ast.If, ast.While, ast.For)):
                    documented_lines.append(f'# {node.__class__.__name__} block')
                    documented_lines.append(line)

                elif isinstance(node, ast.Assign):
                    targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    if targets:
                        documented_lines.append(f'# Assigning to: {", ".join(targets)}')
                    documented_lines.append(line)
                else:
                    documented_lines.append(line)

            return '\n'.join(documented_lines)

        except Exception as e:
            return f"Error generating inline documentation: {str(e)}"

    def _generate_semantic_documentation(self, code: str, analysis: Dict[str, Any]) -> str:
        try:
            code_embedding = self._get_code_embeddings(code)
            similar_docs = self._find_similar_documentation(code_embedding)

            semantic_parts = []
            semantic_parts.append("### Code Complexity Analysis")
            semantic_parts.append(f"- Cyclomatic Complexity: {analysis['complexity']}")
            semantic_parts.append(f"- Number of Functions: {analysis['functions']}")
            semantic_parts.append(f"- Number of Classes: {analysis['classes']}")

            doc = self.nlp(code)
            pos_counts = {}
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

            semantic_parts.append("\n### Part of Speech Analysis")
            for pos, count in pos_counts.items():
                semantic_parts.append(f"- {pos}: {count}")

            if similar_docs:
                semantic_parts.append("\n### Similar Code References")
                for doc in similar_docs[:3]:
                    semantic_parts.append(f"- Similarity Score: {doc['similarity']:.2f}")
                    semantic_parts.append(f"  ```python\n{doc['code'][:200]}...\n  ```")

            return '\n'.join(semantic_parts)

        except Exception as e:
            return f"Error generating semantic documentation: {str(e)}"

    def _generate_prompt_based_documentation(self, code: str, prompt_template: Optional[str] = None) -> str:
        if prompt_template is None:
            prompt_template = COT_FEWSHOT_TEMPLATE
        prompt = prompt_template.format(code=code)
        
        # Tokenize the prompt
        inputs = self.doc_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Generate the documentation using the fine-tuned model
        outputs = self.doc_model.generate(inputs["input_ids"])
        docstring = self.doc_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return docstring

    def _get_code_embeddings(self, code: Optional[str]) -> np.ndarray:
        if code is None:
            code = ""
        inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def _store_documentation(self, code: str, documentation: str):
        embeddings = self._get_code_embeddings(code)
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embeddings_dim)
        # Ensure embeddings is not None and is the correct shape
        if embeddings is not None:
            self.index.add(embeddings.astype(np.float32))  # type: ignore
        doc_id = len(self.documentation_store)
        self.documentation_store[doc_id] = {
            'code': code,
            'documentation': documentation,
            'embeddings': embeddings
        }

    def _find_similar_documentation(self, query_embeddings: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        if self.index is None or not self.documentation_store:
            return []

        # Ensure query_embeddings is not None and has correct shape
        if query_embeddings is None:
            return []
        D, I = self.index.search(query_embeddings.astype(np.float32), k)  # type: ignore
        results = []
        for distance, idx in zip(D[0], I[0]):
            if idx in self.documentation_store:
                similarity = 1 / (1 + distance)
                results.append({
                    'code': self.documentation_store[idx]['code'],
                    'documentation': self.documentation_store[idx]['documentation'],
                    'similarity': similarity
                })
        return results

    def _find_node_at_line(self, tree: ast.AST, line_number: int) -> Optional[ast.AST]:
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and getattr(node, 'lineno', None) == line_number:
                return node
        return None
