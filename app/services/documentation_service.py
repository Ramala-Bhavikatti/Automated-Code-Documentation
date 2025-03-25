import ast
from typing import Dict, Any, List
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import spacy
from app.services.code_service import CodeService

class DocumentationService:
    def __init__(self):
        self.code_service = CodeService()
        self.nlp = spacy.load("en_core_web_sm")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")
        
        # Initialize FAISS index
        self.index = None
        self.documentation_store = {}
        self.embeddings_dim = 768  # CodeBERT embedding dimension
    
    def generate_documentation(self, code: str) -> Dict[str, Any]:
        """Generate documentation for the given code"""
        # Analyze code first
        analysis = self.code_service.analyze_code(code)
        
        # Generate inline documentation
        inline_doc = self._generate_inline_documentation(code, analysis)
        
        # Generate semantic documentation using RAG
        semantic_doc = self._generate_semantic_documentation(code, analysis)
        
        # Combine both documentations
        documentation = f"""# Code Documentation

## Inline Documentation
{inline_doc}

## Semantic Analysis
{semantic_doc}
"""
        
        # Store documentation for future retrieval
        self._store_documentation(code, documentation)
        
        return {
            'documentation': documentation,
            'analysis': analysis
        }
    
    def _generate_inline_documentation(self, code: str, analysis: Dict[str, Any]) -> str:
        """Generate inline documentation using code analysis"""
        try:
            tree = ast.parse(code)
            lines = code.split('\n')
            documented_lines = []
            
            # Add file-level docstring
            if ast.get_docstring(tree):
                documented_lines.append(f'"""\n{ast.get_docstring(tree)}\n"""\n')
            
            # Process each line with its AST node
            for i, line in enumerate(lines):
                node = self._find_node_at_line(tree, i + 1)
                
                if isinstance(node, ast.ClassDef):
                    # Add class docstring
                    if ast.get_docstring(node):
                        documented_lines.append(f'# Class: {node.name}')
                        documented_lines.append(f'# {ast.get_docstring(node)}')
                    else:
                        documented_lines.append(f'# Class: {node.name} - No docstring available')
                    documented_lines.append(line)
                
                elif isinstance(node, ast.FunctionDef):
                    # Add function docstring
                    if ast.get_docstring(node):
                        documented_lines.append(f'# Function: {node.name}')
                        documented_lines.append(f'# {ast.get_docstring(node)}')
                    else:
                        documented_lines.append(f'# Function: {node.name} - No docstring available')
                    documented_lines.append(line)
                
                elif isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                    # Add control structure comments
                    documented_lines.append(f'# {node.__class__.__name__} block')
                    documented_lines.append(line)
                
                elif isinstance(node, ast.Assign):
                    # Add variable assignment comments
                    targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
                    if targets:
                        documented_lines.append(f'# Assigning to: {", ".join(targets)}')
                    documented_lines.append(line)
                
                else:
                    documented_lines.append(line)
            
            return '\n'.join(documented_lines)
            
        except Exception as e:
            return f"Error generating inline documentation: {str(e)}"
    
    def _generate_semantic_documentation(self, code: str, analysis: Dict[str, Any]) -> str:
        """Generate semantic documentation using RAG"""
        try:
            # Get code embeddings
            code_embedding = self._get_code_embeddings(code)
            
            # Find similar code snippets
            similar_docs = self._find_similar_documentation(code_embedding)
            
            # Generate semantic analysis
            semantic_parts = []
            
            # Add code complexity analysis
            semantic_parts.append("### Code Complexity Analysis")
            semantic_parts.append(f"- Cyclomatic Complexity: {analysis['complexity']}")
            semantic_parts.append(f"- Number of Functions: {analysis['functions']}")
            semantic_parts.append(f"- Number of Classes: {analysis['classes']}")
            
            # Add POS analysis
            doc = self.nlp(code)
            semantic_parts.append("\n### Part of Speech Analysis")
            pos_counts = {}
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            for pos, count in pos_counts.items():
                semantic_parts.append(f"- {pos}: {count}")
            
            # Add similar code references
            if similar_docs:
                semantic_parts.append("\n### Similar Code References")
                for doc in similar_docs[:3]:  # Show top 3 similar docs
                    semantic_parts.append(f"- Similarity Score: {doc['similarity']:.2f}")
                    semantic_parts.append(f"  ```python\n{doc['code'][:200]}...\n  ```")
            
            return '\n'.join(semantic_parts)
            
        except Exception as e:
            return f"Error generating semantic documentation: {str(e)}"
    
    def _get_code_embeddings(self, code: str) -> np.ndarray:
        """Get code embeddings using CodeBERT"""
        inputs = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    def _store_documentation(self, code: str, documentation: str):
        """Store documentation in FAISS index"""
        # Get code embeddings
        embeddings = self._get_code_embeddings(code)
        
        # Initialize FAISS index if not exists
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.embeddings_dim)
        
        # Add to index
        self.index.add(embeddings.astype(np.float32))
        
        # Store documentation
        doc_id = len(self.documentation_store)
        self.documentation_store[doc_id] = {
            'code': code,
            'documentation': documentation,
            'embeddings': embeddings
        }
    
    def _find_similar_documentation(self, query_embeddings: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Find similar documentation using FAISS"""
        if self.index is None or not self.documentation_store:
            return []
        
        # Search in FAISS index
        D, I = self.index.search(query_embeddings.astype(np.float32), k)
        
        # Return similar documentation with similarity scores
        results = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx in self.documentation_store:
                similarity = 1 / (1 + distance)  # Convert distance to similarity score
                results.append({
                    'code': self.documentation_store[idx]['code'],
                    'documentation': self.documentation_store[idx]['documentation'],
                    'similarity': similarity
                })
        return results
    
    def _find_node_at_line(self, tree: ast.AST, line_number: int) -> ast.AST:
        """Find the AST node that starts at the given line number"""
        for node in ast.walk(tree):
            if hasattr(node, 'lineno') and node.lineno == line_number:
                return node
        return None 