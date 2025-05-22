import pandas
import numpy as np
from scipy.stats import chi2

df = pandas.read_csv('kc_house_data.csv')


df['age'] = 2025 - df['yr_built']

new_houses = df[df['age'] <=50]['price'].values
old_houses = df[df['age'] > 50]['price'].values


bins =  np.linspace(df['price'].min(), df['price'].max(), 11)

new_counts, _ = np.histogram(new_houses, bins=bins)
old_counts, _ = np.histogram(old_houses, bins=bins)

observed = np.array([new_counts, old_counts])

total_counts = observed.sum(axis=0)
total_new_counts = new_counts.sum()
total_old_counts = old_counts.sum()
total = total_new_counts + total_old_counts

expected_new = total_new_counts * (total_new_counts / total)
expected_old = total_old_counts * (total_old_counts / total)

expected = np.array([expected_new, expected_old])

valid = (expected > 0).all(axis=0)
observed = observed[:, valid]
expected = expected[:, valid]

chi2_stat = np.sum((observed - expected) ** 2 / expected)

dof = observed.shape[0] - 1

p_value = 1 - chi2.cdf(chi2_stat, dof)


print(f"Статистика X^2: {chi2_stat}")
print(f"p-value: {p_value}")
print(f"Степени свободы: {dof}")

if p_value > 0.05:
    print('Выборки можно считать однородными')
else:
    print('Выборки неоднородные')