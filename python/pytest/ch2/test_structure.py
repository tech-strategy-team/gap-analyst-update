from cards import Card

def test_to_dict():
    # GIVEN 前提　基地の値が設定されたCardsオブジェクトが与えられたとすれば
    c1 = Card("somthing", "brian", "todo", 123)

    # WHEN もし　このオブジェクトのto_dictメソッドを呼び出したら
    c2 = c1.to_dict()

    # THEN その結果は　元のオブジェクトの属性を持つ辞書であるべき
    c2_expected = {
        "summary": "somthing",
        "owner": "brian",
        "state": "todo",
        "id": 123,
    }
    assert c2 == c2_expected