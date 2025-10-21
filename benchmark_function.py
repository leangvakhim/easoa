# benchmark_functions.py
import numpy as np

def ackley(x):
    """
    Ackley Function.
    Domain: [-32.768, 32.768]
    Global Minimum: f(x) = 0 at x = [0, ..., 0]
    """
    n = len(x)
    if n == 0:
        return 0.0
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(2 * np.pi * x))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20.0 + np.e

def griewank(x):
    """
    Griewank Function.
    Domain: [-600, 600]
    Global Minimum: f(x) = 0 at x = [0, ..., 0]
    """
    if len(x) == 0:
        return 0.0
    sum_term = np.sum(x**2 / 4000.0)

    # Create an array [1, 2, ..., n]
    i = np.arange(1, len(x) + 1)
    prod_term = np.prod(np.cos(x / np.sqrt(i)))

    return sum_term - prod_term + 1.0

def schwefel_1_2(x):
    """
    Schwefel's Problem 1.2 (Rotated Hyper-Ellipsoid).
    Domain: [-100, 100]
    Global Minimum: f(x) = 0 at x = [0, ..., 0]
    """
    n = len(x)
    if n == 0:
        return 0.0
    total_sum = 0.0
    for i in range(n):
        inner_sum = np.sum(x[:i+1])
        total_sum += inner_sum**2
    return total_sum

def high_conditioned_elliptic(x):
    """
    High Conditioned Elliptic Function.
    Domain: [-100, 100]
    Global Minimum: f(x) = 0 at x = [0, ..., 0]
    """
    n = len(x)
    if n == 0:
        return 0.0

    # Create coefficients [10^0, 10^6(1/(n-1)), 10^6(2/(n-1)), ...]
    i = np.arange(n)
    exponents = 6 * (i / (n - 1))
    coeffs = 10**exponents

    return np.sum(coeffs * x**2)