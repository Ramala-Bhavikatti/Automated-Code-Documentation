# app/utils/parsers.py
import fitz  # PyMuPDF
from PIL import Image


def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        text = f"[PDF Extraction Failed] {str(e)}"
    return text.strip()


import easyocr

reader = easyocr.Reader(['en'], gpu=False)  # Load English model

def extract_text_from_image(image_path: str) -> str:
    try:
        result = reader.readtext(image_path, detail=0)
        return "\n".join(result).strip()
    except Exception as e:
        return f"[Image Extraction Failed] {str(e)}"

