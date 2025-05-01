from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Programmer(Base):
    __tablename__ = "programmer"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False, unique=True)
    twitter_id: Mapped[str] = mapped_column(nullable=False, unique=True)
    languages = relationship(
        "ProgrammerLanguage",
        backref="programmer",
        cascade="all, delete-orphan",
    )


class ProgrammerLanguage(Base):
    __tablename__ = "programmer_language"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(nullable=False)
    programmer_id: Mapped[int] = mapped_column(
        ForeignKey("programmer.id"), nullable=False
    )
