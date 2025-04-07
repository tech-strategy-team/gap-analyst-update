import cards

def test_version(capsys):
    """Test the version of the cards module"""
    cards.cli.version()
    output = capsys.readouterr().out.strip()
    assert output == cards.__version__
