from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

class CodeSearchNetIndexer:
    def __init__(self, language="python", limit=1000):
        self.language = language
        self.limit = limit
        self.model = SentenceTransformer('microsoft/codebert-base')
        self.index_path = "vectorstore/code_index.faiss"
        self.snippet_path = "vectorstore/snippets.npy"

    def load_data(self):
        print("📥 Loading CodeSearchNet dataset...")
        dataset = load_dataset("code_search_net", self.language, split=f"train[:{self.limit}]", trust_remote_code=True)

        if len(dataset) > 0:
            print(f"🔎 Sample row keys: {dataset[0].keys()}")

        code_snippets = [row["func_code_string"] for row in dataset if row.get("func_code_string")]

        print(f"✅ Loaded {len(code_snippets)} code snippets.")
        return code_snippets

    def build_index(self, code_snippets):
        if not code_snippets:
            print("⚠️ No code snippets to index. Exiting.")
            return

        os.makedirs("vectorstore", exist_ok=True)

        print("🔍 Generating embeddings...")
        embeddings = self.model.encode(code_snippets, convert_to_numpy=True)

        print("📦 Building FAISS index...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        print("💾 Saving index and snippets...")
        faiss.write_index(index, self.index_path)
        np.save(self.snippet_path, code_snippets)

        print("✅ Done! Index and snippets saved.")

if __name__ == "__main__":
    indexer = CodeSearchNetIndexer(language="python", limit=1000)
    snippets = indexer.load_data()
    indexer.build_index(snippets)
