from functools import wraps
from flask import request, jsonify
from jose import JWTError, jwt
from datetime import datetime, timedelta
import os

def create_access_token(data: dict):
    """Create a new JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 30)))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, os.getenv('JWT_SECRET_KEY', 'dev'), algorithm=os.getenv('JWT_ALGORITHM', 'HS256'))
    return encoded_jwt

def token_required(f):
    """Decorator to require JWT token for protected routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            token = request.headers['Authorization'].split(" ")[1]
        
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            data = jwt.decode(token, os.getenv('JWT_SECRET_KEY', 'dev'), algorithms=[os.getenv('JWT_ALGORITHM', 'HS256')])
            current_user = data
        except JWTError:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated

def generate_token(user_id: str) -> str:
    """Generate a new JWT token"""
    expires_delta = timedelta(minutes=int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES', 30)))
    expire = datetime.utcnow() + expires_delta
    
    to_encode = {
        'sub': user_id,
        'exp': expire
    }
    
    encoded_jwt = jwt.encode(
        to_encode,
        os.getenv('JWT_SECRET_KEY'),
        algorithm=os.getenv('JWT_ALGORITHM')
    )
    
    return encoded_jwt 