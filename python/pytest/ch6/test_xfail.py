import pytest
import cards
from cards import Card
from packaging.version import parse

@pytest.mark.xfail(
    parse(cards.__version__).major < 2,
    reason="This test is expected to fail"
)
def test_less_than():
    c1 = Card("A")
    c2 = Card("B")
    assert c1 < c2

@pytest.mark.xfail(reason="XPASS demo")
def test_xpass():
    c1 = Card("A")
    c2 = Card("A")
    assert c1 == c2


@pytest.mark.xfail(reason="strict demo", strict=True)
def test_xfail_strict():
    c1 = Card("A")
    c2 = Card("A")
    assert c1 == c2