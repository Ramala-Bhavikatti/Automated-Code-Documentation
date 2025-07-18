import faiss
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModel
import torch


class CodeEmbeddingStore:
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # Set model to inference mode

        self.index = faiss.IndexFlatL2(768)  # CodeBERT embedding dimension
        self.snippets: List[str] = []

    def embed_code(self, code: str) -> np.ndarray:
        """
        Converts code snippet to an embedding using CodeBERT.
        """
        inputs = self.tokenizer(code, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean pooling over token embeddings
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding

    def add_code_snippet(self, code: str):
        """
        Embeds and adds a code snippet to FAISS index and in-memory list.
        """
        embedding = self.embed_code(code)
        self.index.add(np.expand_dims(embedding, axis=0))
        self.snippets.append(code)

    def save_index(self, index_path: str = "vectorstore/code_index.faiss", snippets_path: str = "vectorstore/snippets.npy"):
        """
        Save FAISS index and corresponding code snippets to disk.
        """
        faiss.write_index(self.index, index_path)
        np.save(snippets_path, np.array(self.snippets, dtype=object))  # Save list of strings

    def load_index(self, index_path: str = "vectorstore/code_index.faiss", snippets_path: str = "vectorstore/snippets.npy"):
        """
        Load FAISS index and code snippets from disk.
        """
        self.index = faiss.read_index(index_path)
        self.snippets = np.load(snippets_path, allow_pickle=True).tolist()

    def search_similar_code(self, query_code: str, k: int = 3) -> List[str]:
        if self.index.ntotal == 0:
            return []

        query_embedding = self.embed_code(query_code).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i in indices[0]:
            if i < len(self.snippets):
                results.append(self.snippets[i])
        return results
