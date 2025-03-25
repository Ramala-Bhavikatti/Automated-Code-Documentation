"""
Real-Time Collaborative Code Documentation
"""

from flask import Flask
from flask_socketio import SocketIO
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'dev')
app.config['DEBUG'] = os.getenv('DEBUG', 'True').lower() == 'true'

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*")

# Import routes after app initialization to avoid circular imports
from app.api import routes

# Register blueprints
app.register_blueprint(routes.api)

# Register socket events
routes.register_socket_events(socketio) 