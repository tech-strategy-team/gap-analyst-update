from cards import Card

def test_filed_access():
    c = Card("Somthing", "brian", "todo", 123)
    assert c.summary == "Somthing"
    assert c.owner == "brian"
    assert c.state == "todo"
    assert c.id == 123

def test_defaults():
    c = Card()
    assert c.summary is None
    assert c.owner is None
    assert c.state == "todo"
    assert c.id is None

def test_equality():
    c1 = Card("Somthing", "brian", "todo", 123)
    c2 = Card("Somthing", "brian", "todo", 123)
    assert c1 == c2

def test_equality_wtih_diff_ids():
    c1 = Card("Somthing", "brian", "todo", 123)
    c2 = Card("Somthing", "brian", "todo", 456)
    assert c1 == c2

def test_inequality():
    c1 = Card("Somthing", "brian", "todo", 123)
    c2 = Card("Somthing else", "elon", "done", 123)
    assert c1 != c2

def test_from_dict():
    c1 = Card("Somthing", "brian", "todo", 123)
    c2_dict = {
        "summary": "Somthing",
        "owner": "brian",
        "state": "todo",
        "id": 123
    }
    c2 = Card.from_dict(c2_dict)
    assert c1 == c2

def test_to_dict():
    c1 = Card("Somthing", "brian", "todo", 123)
    c2 = c1.to_dict()
    c2_expected = {
        "summary": "Somthing",
        "owner": "brian",
        "state": "todo",
        "id": 123
    }
    assert c2 == c2_expected
