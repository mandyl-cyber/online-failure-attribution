def solve(values, size):
    if size <= 0:
        raise ValueError('Size must be greater than 0')
    return [values[i:i+size] for i in range(0, len(values) - size + 1, size)]
