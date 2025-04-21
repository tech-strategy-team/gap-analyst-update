from cards import Card

def test_finish_from_in_prog(cards_db):
    # このテストは、"in prog" 状態のカードを追加し、finishメソッドを呼び出すことで、カードの状態が "done" に変更されることを確認するためのものです。
    
    # "in prog" 状態のカードをカードデータベースに追加
    index = cards_db.add_card(Card("second eddition", state="in prog"))
    # カードの状態を "done" に変更
    cards_db.finish(index)
    card = cards_db.get_card(index)
    # カードの状態が "done" に変更されていることを確認
    assert card.state == "done"


def test_finish_from_done(cards_db):
    # このテストは、"done" 状態のカードを追加し、finishメソッドを呼び出しても状態が変更されないことを確認するためのものです。
    index = cards_db.add_card(Card("write a book", state="done"))
    cards_db.finish(index)
    card = cards_db.get_card(index)
    # カードの状態が "done" のままであることを確認
    assert card.state == "done"


def test_finish_from_todo(cards_db):
    # このテストは、"todo" 状態のカードを追加し、finishメソッドを呼び出すことで、カードの状態が "done" に変更されることを確認するためのものです。
    
    # "todo" 状態のカードをカードデータベースに追加
    index = cards_db.add_card(Card("write a book", state="todo"))
    # カードの状態を "done" に変更
    cards_db.finish(index)
    card = cards_db.get_card(index)
    # カードの状態が "done" に変更されていることを確認
    assert card.state == "done"