from cards import Card

def test_finish(cards_db):
    for c in [
        Card("write book", "Brian", "done"),
        Card("edit book", "Katie", "in_prog"),
        Card("write 2nd edition", "Brian", "todo")
    ]:
        index = cards_db.add_card(c)
        cards_db.finish(index)
        card = cards_db.get_card(index)
        assert card.state == "done"

