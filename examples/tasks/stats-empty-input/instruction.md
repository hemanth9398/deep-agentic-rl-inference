Calling `stats.mean([])` raises `ZeroDivisionError`, and `stats.median([])` raises
`IndexError`. Both should return `0.0` for an empty input instead of crashing. Behaviour for
non-empty inputs must not change.
