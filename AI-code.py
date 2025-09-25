def fibonacci_iterative(n):
    """Generates the Fibonacci sequence up to n terms iteratively."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    else:
        sequence = [0, 1]
        while len(sequence) < n:
            next_term = sequence[-1] + sequence[-2]
            sequence.append(next_term)
        return sequence

# Example usage:
print(fibonacci_iterative(10))