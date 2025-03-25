from sqlalchemy import Column, String, Text, ForeignKey, Integer, JSON
from sqlalchemy.orm import relationship
from app.models.base import BaseModel

class Documentation(BaseModel):
    """Model for storing code documentation"""
    __tablename__ = 'documentation'
    
    title = Column(String(200), nullable=False)
    code = Column(Text, nullable=False)
    documentation = Column(Text, nullable=False)
    analysis = Column(JSON)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    # Relationships
    user = relationship('User', backref='documentation')
    
    def to_dict(self):
        """Convert documentation to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'code': self.code,
            'documentation': self.documentation,
            'analysis': self.analysis,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        } 