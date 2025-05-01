from pydantic import BaseModel


class ProgrammerListItem(BaseModel):
    """一覧用プログラマー情報"""
    name: str


class ProgrammerDetail(BaseModel):
    """プログラマー情報"""
    name: str
    twitter_id: str
    language: list[str]
