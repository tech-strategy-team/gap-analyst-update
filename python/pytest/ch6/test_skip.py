from cards import Card
import pytest

@pytest.mark.skip(reason="Not implemented yet")
def test_less_than():
    c1 = Card("test a")
    c2 = Card("test b")
    assert c1 < c2

def test_equality():
    c1 = Card("test a")
    c2 = Card("test a")
    assert c1 == c2