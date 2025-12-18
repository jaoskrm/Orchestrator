from main import is_prime

def test_small():
    assert is_prime(2) is True
    assert is_prime(3) is True
    assert is_prime(4) is False

def test_edge():
    assert is_prime(1) is False
    assert is_prime(0) is False
    assert is_prime(-7) is False

def test_bigger():
    assert is_prime(29) is True
    assert is_prime(49) is False
