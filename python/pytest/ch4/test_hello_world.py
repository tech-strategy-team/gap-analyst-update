import pytest
from hello_world import hello  # 修正: hello関数を直接インポート

def test_hello():
        # Test the hello_world function
    hello()

    # Check if the file was created
    with open("hello.txt", "r") as f:
        content = f.read()
    
    assert content == "Hello World!\n"

def test_hello2(monkeypatch, tmp_path, capsys):
    # 一時ディレクトリに変更
    monkeypatch.chdir(tmp_path)

    # Test the hello_world function
    hello()
    with capsys.disabled():
        print(tmp_path)

    # Check if the file was created
    with open("hello.txt", "r") as f:
        content = f.read()
    
    assert content == "Hello World!\n"