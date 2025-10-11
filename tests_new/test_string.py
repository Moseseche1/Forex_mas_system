def test_uppercase():
    assert "hello".upper() == "HELLO"

def test_split():
    s = "one two three"
    assert s.split() == ["one", "two", "three"]
