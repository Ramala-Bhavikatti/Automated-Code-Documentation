from flask import Blueprint, request, jsonify
from flask_socketio import emit, join_room
from app.services.code_service import CodeService
from app.services.documentation_service import DocumentationService
from app.utils.auth import token_required, create_access_token

api = Blueprint('api', __name__)
code_service = CodeService()
doc_service = DocumentationService()

@api.route('/api/auth/token', methods=['POST'])
def get_token():
    """Get a JWT token for authentication"""
    data = request.get_json()
    if not data or 'username' not in data:
        return jsonify({'error': 'Username is required'}), 400
    
    token = create_access_token({'username': data['username']})
    return jsonify({'token': token})

@api.route('/api/code/document', methods=['POST'])
@token_required
def document_code(current_user):
    """Generate documentation for code"""
    data = request.get_json()
    code = data.get('code')
    if not code:
        return jsonify({'error': 'No code provided'}), 400
    
    try:
        result = doc_service.generate_documentation(code)
        return jsonify({'documentation': result['documentation']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/api/code/analyze', methods=['POST'])
@token_required
def analyze_code(current_user):
    """Analyze code structure and complexity"""
    data = request.get_json()
    code = data.get('code')
    if not code:
        return jsonify({'error': 'No code provided'}), 400
    
    try:
        analysis = code_service.analyze_code(code)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket events
def register_socket_events(socketio):
    @socketio.on('join_room')
    def on_join(data):
        room = data['room']
        join_room(room)
        emit('status', {'msg': f'User joined room {room}'}, room=room)

    @socketio.on('code_update')
    def on_code_update(data):
        room = data['room']
        code = data['code']
        emit('code_update', {'code': code}, room=room, include_self=False)

    @socketio.on('documentation_update')
    def on_documentation_update(data):
        room = data['room']
        documentation = data['documentation']
        emit('documentation_update', {'documentation': documentation}, room=room, include_self=False) 