def solve(values):
    total = 0
    result = [0]  # Add initial zero
    for i in range(1, len(values)):
        total += values[i-1]
        result.append(total)
    return result
    # Added comment for clarity
    # Added comment for clarity
    # Added comment for clarity
