def mean(values):
    """Arithmetic mean of a sequence of numbers."""
    return sum(values) / len(values)


def median(values):
    """Median of a sequence of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
