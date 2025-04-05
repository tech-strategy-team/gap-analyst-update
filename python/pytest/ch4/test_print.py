def test_normal():
    print("Hello, pytest!")

def test_fail():
    print("This test will fail")
    assert False

def test_disabled(capsys):
    with capsys.disabled():
        print("\nThis will not be captured")
