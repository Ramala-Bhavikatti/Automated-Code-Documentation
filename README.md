# Real-Time Collaborative Code Documentation

An AI-powered collaborative code documentation system that automatically generates and manages code documentation with real-time collaboration features.

## Features

- 🤖 AI-powered code documentation generation
- 👥 Real-time collaborative editing
- 🎤 Multi-modal input support (text, voice, images)
- 🔍 Advanced NLP and RAG capabilities
- 📊 Evaluation framework for documentation quality

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Initialize the database:
```bash
python scripts/init_db.py
```

5. Run the development server:
```bash
python app.py
```

## Project Structure

```
├── app/
│   ├── api/            # API endpoints
│   ├── models/         # Database models
│   ├── services/       # Business logic
│   ├── utils/          # Utility functions
│   └── templates/      # Frontend templates
├── config/             # Configuration files
├── data/              # Dataset storage
├── models/            # ML model storage
├── scripts/           # Utility scripts
├── tests/             # Test files
├── app.py             # Main application entry
└── requirements.txt   # Project dependencies
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 