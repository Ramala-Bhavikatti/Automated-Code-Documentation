from typing import Dict, Optional
from app.services.doc_rag_service import CodeEmbeddingStore
from app.services.code_service import CodeService
from app.services.documentation_service import DocumentationService
from app.utils.parsers import extract_text_from_pdf, extract_text_from_image

class AgenticWorkflow:
    def __init__(self):
        self.code_service = CodeService()
        self.doc_service = DocumentationService()
        self.rag_store = CodeEmbeddingStore()

        try:
            self.rag_store.load_index()
        except Exception as e:
            print(f"[Warning] Could not load FAISS index: {e}")

    def run(self, code: Optional[str] = "", file_path: Optional[str] = None) -> Dict:
        # Extract code/text from file if code is empty
        if not code.strip() and file_path:
            if file_path.lower().endswith('.pdf'):
                code = extract_text_from_pdf(file_path)
            elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                code = extract_text_from_image(file_path)

        if not code.strip():
            raise ValueError("No valid code or extractable content provided.")

        analysis = self.code_service.analyze_code(code)
        similar_docs = self.rag_store.search_similar_code(code)
        context = "\n\n".join(similar_docs) if similar_docs else "None"
        documentation_data = self.doc_service.generate_documentation(code)

        return {
            "analysis": analysis,
            "documentation": documentation_data["documentation"]
        }
