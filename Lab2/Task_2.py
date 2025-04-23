import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from githublabs.MathStat.Lab1.Lab1_3 import theta_estimates

lambdaT = 1
theta = lambdaT + lambdaT ** 2
alpha = 0.05
q = norm.ppf(1 - alpha / 2)
theta_estimates = []

def simulate_coverage(n):
    covered = 0
    lengths = []
    intervals = []

    for _ in range(1000):
        sample = np.random.poisson(lam=lambdaT, size=n)

        X2 = sample**2
        theta_hat = np.mean(X2)

        s = np.std(X2, ddof=1)
        std_error = s / np.sqrt(n)

        ci_lower = theta_hat - q * std_error
        ci_upper = theta_hat + q * std_error

        theta_estimates.append(theta_hat)
        intervals.append((ci_lower, ci_upper))
        lengths.append(ci_upper - ci_lower)

        if ci_lower <= theta <= ci_upper:
            covered += 1

    coverage = covered / 1000
    avg_length = np.mean(lengths)
    plt.boxplot(theta_estimates)
    plt.axhline(theta, color="blue", linestyle='-.', label='Истинное значение E[x^2]')
    plt.title(f"Boxplot оценок E[X^2] для выборки n={n}")
    plt.legend()
    plt.show()
    return coverage, avg_length


coverage, avg_length = simulate_coverage(25)
print(f"Для выборки (n=25) Покрытие: {coverage:.3f}, Средняя длина интервала: {avg_length:.3f}")

coverage, avg_length = simulate_coverage(10000)
print(f"Для выборки (n=10000) Покрытие: {coverage:.3f}, Средняя длина интервала: {avg_length:.3f}")
