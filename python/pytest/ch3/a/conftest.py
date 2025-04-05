from pathlib import Path
from tempfile import TemporaryDirectory
import cards
import pytest


@pytest.fixture(scope="session")
def db():
    """CardsDB object connected to a temporary database"""
    with TemporaryDirectory() as db_dir:
        db_path = Path(db_dir)
        db = cards.CardsDB(db_path)
        yield db
        db.close()


@pytest.fixture(scope="function")
def cards_db(db):
    db.delete_all()
    return db