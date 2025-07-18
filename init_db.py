from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models.base import Base
from app.models.user import User
from app.models.documentation import Documentation
import os
from dotenv import load_dotenv

def init_database():
    """Initialize the database with required tables"""
    load_dotenv()
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    # Create database engine
    engine = create_engine(database_url)
    
    # Create all tables
    Base.metadata.create_all(engine)
    
    # Create session
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Create admin user if not exists
        admin = session.query(User).filter_by(email="admin@example.com").first()
        if not admin:
            admin = User(
                email="admin@example.com",
                username="admin",
                is_admin=True
            )
            admin.set_password("admin123")  # Change this in production
            session.add(admin)
            session.commit()
            print("Created admin user")
        
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    init_database() 