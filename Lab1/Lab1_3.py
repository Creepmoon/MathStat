import numpy as np
import matplotlib.pyplot as plt

theta_0 = 30.6634

sample_sizes = np.array([10, 50, 100, 500, 1000])

M = 1338


def estimate_theta(samples):
    return np.mean(samples)


theta_estimates = {n: [] for n in sample_sizes}

for n in sample_sizes:
    for _ in range(M):
        sample = np.random.uniform( theta_0 - 6.0959, 6.0959 + theta_0, n)
        theta_estimates[n].append(estimate_theta(sample))

fig, axes = plt.subplots(2, len(sample_sizes), figsize=(15, 6))

for i, n in enumerate(sample_sizes):
    axes[0, i].hist(theta_estimates[n], bins=20, density=True, alpha=0.7)
    axes[0, i].axvline(theta_0, color='r', linestyle='dashed', label="θ₀")
    axes[0, i].set_title(f"Гистограмма, n={n}")

    axes[1, i].boxplot(theta_estimates[n], vert=True, widths=0.6)
    axes[1, i].axhline(theta_0, color='r', linestyle='dashed', label="θ₀")
    axes[1, i].set_title(f"Box-plot, n={n}")

plt.tight_layout()
plt.show()