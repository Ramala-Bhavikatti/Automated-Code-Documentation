from setuptools import setup, find_packages

# Read README.md with UTF-8 encoding
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="code-docs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask==2.0.1",
        "flask-socketio==5.1.1",
        "transformers==4.30.2",
        "torch>=2.2.0",
        "datasets==2.12.0",
        "faiss-cpu==1.7.4",
        "nltk==3.8.1",
        "spacy==3.5.3",
        "python-dotenv==1.0.0",
        "openai==0.27.8",
        "pytesseract==0.3.10",
        "Pillow==9.5.0",
        "sacrebleu==2.3.1",
        "SQLAlchemy==2.0.15",
        "python-jose==3.3.0",
        "passlib==1.7.4",
        "python-multipart==0.0.6",
        "werkzeug>=3.0.0"
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Real-Time Collaborative Code Documentation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/code-docs",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Documentation",
    ],
) 