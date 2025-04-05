import pytest


@pytest.fixture(name="ultimate_answer")
def ultimate_answer_fixture():
    """Ultimate answer to the Ultimate Question of Life, the Universe, and Everything."""
    return 42


def test_everything(ultimate_answer):
    """Test everything."""
    assert ultimate_answer == 42