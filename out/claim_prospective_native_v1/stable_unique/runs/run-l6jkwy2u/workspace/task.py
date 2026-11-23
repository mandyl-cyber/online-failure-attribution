def solve(values):
    seen = {}
    result = []
    for value in values:
        if value not in seen:
            seen[value] = True
            result.append(value)
    return result
