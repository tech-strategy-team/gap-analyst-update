from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


Engine = create_engine(
    "sqlite:///sample_db.sqlite3", echo=False
)


SessionLocal = sessionmaker(bind=Engine)
