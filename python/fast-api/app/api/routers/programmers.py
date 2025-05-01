from fastapi import APIRouter, Depends
from app.api.dependencies import get_db
from app.api import cruds
from app.api.schemas import (
    ProgrammerListItem,
    ProgrammerDetail
)


router = APIRouter()


@router.get(
    "/",
    response_model=list[ProgrammerListItem],
)
def list_programmers(
    db = Depends(get_db)
):
    """プログラマー一覧を取得する"""
    return cruds.get_programmers(db)


@router.get(
    "/{name}",
    response_model=ProgrammerDetail,
)
def detail_programmer(name: str):
    # TODO: 実装：データの取得
    return ProgrammerDetail(
        name="susumuis",
        twitter_id="susumuis",
        language=["Python", "JavaScript"],
    )


@router.post(
    "/",
)
def add_programmer(
    programmer: ProgrammerDetail,
    db=Depends(get_db),
):
    cruds.add_programmer(db, programmer)
    return {"result": "ok"}


@router.put(
    "/{name}",
)
def update_programmer(name: str, programmer: ProgrammerDetail):
    # TODO 実装：データの更新
    return {"result": "ok"}


@router.delete(
    "/{name}",
)
def delete_programmer(name: str):
    # TODO 実装：データの削除
    return {"result": "ok"}
