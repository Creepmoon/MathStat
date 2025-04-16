
import numpy as np
from scipy.stats import norm


mu1, sigma1_sq = 2, 1
mu2, sigma2_sq = 1, 0.5
alpha = 0.05
tau = mu1 - mu2
q = norm.ppf(1 - alpha / 2)

def simulate_coverage(n):
    covered = 0
    for _ in range(1000):

        x1 = np.random.normal(mu1, np.sqrt(sigma1_sq), n)
        x2 = np.random.normal(mu2, np.sqrt(sigma2_sq), n)

        mean_diff = np.mean(x1) - np.mean(x2)
        std_error = np.sqrt(sigma1_sq / n + sigma2_sq / n)
        ci_lower = mean_diff - q * std_error
        ci_upper = mean_diff + q * std_error
        if ci_lower <= tau <= ci_upper:
            covered += 1
    return covered / 1000, std_error*2


coverage_25, length = simulate_coverage(25)
print(f"Доля покрытий (n=25): {coverage_25}, длинна интервала {length}")

coverage_10000, length = simulate_coverage(10000)
print(f"Доля покрытий (n=10000): {coverage_10000}, длинна интервала {length}")
