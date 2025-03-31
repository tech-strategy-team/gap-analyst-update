from cards import Card

class TestEquality:
    def test_equality(self):
        c1 = Card("somthing", "brian", "todo", 123)
        c2 = Card("somthing", "brian", "todo", 123)
        assert c1 == c2

    def test_equality_with_diff_ids(self):
        c1 = Card("somthing", "brian", "todo", 123)
        c2 = Card("somthing", "brian", "todo", 456)
        assert c1 == c2

    def test_inequality(self):
        c1 = Card("somthing", "brian", "todo", 123)
        c2 = Card("comletetly", "elon", "done", 123)
        assert c1 != c2