from stats import mean, median


def test_mean_basic():
    assert mean([1, 2, 3]) == 2


def test_median_odd():
    assert median([3, 1, 2]) == 2


def test_mean_empty_returns_zero():
    assert mean([]) == 0.0


def test_median_empty_returns_zero():
    assert median([]) == 0.0
