import pandas as pd
import numpy as np
from scipy.stats import f
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("iris.csv")
df["Area"] = df["Sepal.Length"] * df["Sepal.Width"] + df["Petal.Length"] * df["Petal.Width"]

groups = df.groupby("Species")["Area"]

grand_mean = df["Area"].mean()


ss_between = 0
ss_within = 0
n_total = 0

for name, group in groups:
    n_i = len(group)
    mean_i = group.mean()
    ss_between += n_i * (mean_i - grand_mean) ** 2
    ss_within += np.sum((group - mean_i) ** 2)
    n_total += n_i

ss_total = ss_between + ss_within

k = groups.ngroups
df_between = k - 1
df_within = n_total - k

ms_between = ss_between / df_between
ms_within = ss_within / df_within

f_stat = ms_between / ms_within

p_value = 1 - f.cdf(f_stat, df_between, df_within)

group_means = df.groupby("Species")["Area"].mean()

print(f"F-статистика: {f_stat:.4f}")
print(f"p-value: {p_value:.10f}")
print("\nСредние площади по видам:")
for name, mean in group_means.items():
    print(f"{name:12s}: {mean:.4f}")

plt.figure(figsize=(8, 5))
sns.boxplot(x='Species', y='Area', data=df)
plt.title("Сравнение площади по видам (Boxplot)")
plt.ylabel("Площадь (Area)")
plt.show()