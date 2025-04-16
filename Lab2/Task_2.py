import numpy as np
from scipy.stats import norm

lambdaT = 1
theta = lambdaT + lambdaT ** 2
alpha = 0.05
q = norm.ppf(1 - alpha / 2)


def simulate_coverage(n):
    covered = 0
    lengths = []
    intervals = []

    for _ in range(1000):
        sample = np.random.poisson(lam=lambdaT, size=n)

        lambda_hat = np.mean(sample)
        theta_hat = lambda_hat + lambda_hat ** 2

        var_theta_hat = ((1 + 2 * lambda_hat) ** 2) * lambda_hat / n
        std_error = np.sqrt(var_theta_hat)

        ci_lower = theta_hat - q * std_error
        ci_upper = theta_hat + q * std_error

        intervals.append((ci_lower, ci_upper))
        lengths.append(ci_upper - ci_lower)

        if ci_lower <= theta <= ci_upper:
            covered += 1

    coverage = covered / 1000
    avg_length = np.mean(lengths)

    return coverage, avg_length, intervals


coverage, avg_length, intervals = simulate_coverage(25)
print(f"Для выборки (n=25) Покрытие: {coverage:.3f}, Средняя длина интервала: {avg_length:.3f}, интервалы: {intervals}")

coverage, avg_length, intervals = simulate_coverage(10000)
print(
    f"Для выборки (n=10000) Покрытие: {coverage:.3f}, Средняя длина интервала: {avg_length:.3f}, интервалы: {intervals}")
