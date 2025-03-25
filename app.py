from app import app, socketio
from flask import render_template
import os

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    host = os.getenv('WEBSOCKET_HOST', '0.0.0.0')
    port = int(os.getenv('WEBSOCKET_PORT', 5000))
    socketio.run(app, host=host, port=port, debug=app.config['DEBUG']) 