from sqlalchemy import Column, String, Boolean
from werkzeug.security import generate_password_hash, check_password_hash
from app.models.base import BaseModel

class User(BaseModel):
    """User model for authentication and user management"""
    __tablename__ = 'users'
    
    email = Column(String(120), unique=True, nullable=False)
    username = Column(String(80), unique=True, nullable=False)
    password_hash = Column(String(128))
    is_admin = Column(Boolean, default=False)
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert user to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        } 