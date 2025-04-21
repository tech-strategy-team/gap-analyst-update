import pytest
from cards import Card

@pytest.mark.parametrize(
    "start_state",
    [
        "done",
        "in_prog",
        "todo",
    ],
)


def test_finish(cards_db, start_state):

    initial_card = Card("test",state=start_state)
    index = cards_db.add_card(initial_card)

    cards_db.finish(index)
    card = cards_db.get_card(index)
    assert card.state == "done"

def test_start_from_done(cards_db):

    index = cards_db.add_card(Card("test",state="done"))
    cards_db.start(index)
    card = cards_db.get_card(index)
    assert card.state == "in prog"

def test_start_from_in_prog(cards_db):
    index = cards_db.add_card(Card("test",state="in prog"))
    cards_db.start(index)
    card = cards_db.get_card(index)
    assert card.state == "in prog"

def test_start_from_todo(cards_db):
    index = cards_db.add_card(Card("test",state="todo"))
    cards_db.start(index)
    card = cards_db.get_card(index)
    assert card.state == "in prog"


@pytest.mark.parametrize(
    "start_state",
    [
        "done",
        "in_prog",
        "todo",
    ],
)

def test_start(cards_db, start_state):
    initial_card = Card("test",state=start_state)
    index = cards_db.add_card(initial_card)

    cards_db.start(index)
    card = cards_db.get_card(index)
    assert card.state == "in prog"


@pytest.fixture(params=["done", "in prog", "todo"])
def start_state(request):
    """テストの前に実行されるフィクスチャ"""
    return request.param

def test_start2(cards_db, start_state):
    initial_card = Card("test",state=start_state)
    index = cards_db.add_card(initial_card)

    cards_db.start(index)
    card = cards_db.get_card(index)
    assert card.state == "in prog"

def pytest_generate_tests(metafunc):
    if "start_state2" in metafunc.fixturenames:
        metafunc.parametrize("start_state2", ["done", "in_prog", "todo"])

def test_start3(cards_db, start_state2):
    initial_card = Card("test",state=start_state2)
    index = cards_db.add_card(initial_card)

    cards_db.start(index)
    card = cards_db.get_card(index)
    assert card.state == "in prog"