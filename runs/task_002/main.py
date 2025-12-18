def is_prime(n):
    if n <= 1:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

# Test cases for is_prime function
def test_is_prime():
    assert is_prime(2) == True, "2 should be prime"
    assert is_prime(3) == True, "3 should be prime"
    assert is_prime(4) == False, "4 should not be prime"
    assert is_prime(5) == True, "5 should be prime"
    assert is_prime(17) == True, "17 should be prime"
    assert is_prime(18) == False, "18 should not be prime"
    assert is_prime(19) == True, "19 should be prime"
    assert is_prime(20) == False, "20 should not be prime"

if __name__ == "__main__":
    test_is_prime()
