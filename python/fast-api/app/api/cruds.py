from sqlalchemy import select
from sqlalchemy.orm import Session
from app.api.schemas import ProgrammerDetail
from app.db.models import Programmer, ProgrammerLanguage


def get_programmers(
    db: Session
) -> list[Programmer]:
    """一覧を返す"""
    programmers = db.scalars(
        select(
            Programmer,
        )
        .order_by("id")
    ).all()
    return list(programmers)


def add_programmer(
    db: Session,
    programmer_detail: ProgrammerDetail
):
    """新規登録"""
    programmer = Programmer()
    programmer.name = programmer_detail.name
    programmer.twitter_id = programmer_detail.twitter_id
    programmer.languages = [
        ProgrammerLanguage(name=language)
        for language in programmer_detail.languages
    ]
    db.add(programmer)
    db.commit()
