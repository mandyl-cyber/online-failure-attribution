def solve(base, overrides):
    result = base.copy()
    for k, v in overrides.items():
        if v is not None:
            result[k] = v
    return result
