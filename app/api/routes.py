from flask import request, jsonify, render_template
from werkzeug.utils import secure_filename
import os

from app.services.agent_controller import AgenticWorkflow
from app.utils.parsers import extract_text_from_pdf, extract_text_from_image  # <-- Corrected import

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'py'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

agent = AgenticWorkflow()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def register_routes(app):

    @app.route('/')
    def index():
        return render_template("index.html")


    # Agentic Workflow
    @app.route('/agentic-workflow', methods=['POST'])
    def run_agentic_workflow():
        print("Received POST to /agentic-workflow")
        code = request.form.get("code", "")
        print(f"🧠 Received code: {code[:100]}")
        file = request.files.get("file")
        file_path = None

        
        if file and file.filename and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            ext = filename.rsplit(".", 1)[1].lower()
            if ext == "pdf":
                extracted_text = extract_text_from_pdf(file_path)
                code = code or extracted_text
            elif ext in {"png", "jpg", "jpeg"}:
                extracted_text = extract_text_from_image(file_path)
                code = code or extracted_text
            elif ext == "py":
                with open(file_path, "r", encoding="utf-8") as f:
                    code = code or f.read()

        try:
            result = agent.run(code=code, file_path=file_path)
            return jsonify(result)
        except Exception as e:
            print("Agentic workflow error:", e)
            return jsonify({"error": str(e)}), 400