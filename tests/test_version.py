from src.main import testingtests


def test_a():
    input_val = 2
    # the function simply outputs input_val + 1
    a = testingtests.test_a(a=input_val)
    assert a == input_val + 1
