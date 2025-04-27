import pandas
import numpy as np
from scipy.special import obl_ang1
from scipy.stats import chi2_contingency

df = pandas.read_csv('kc_house_data.csv')


df['age'] = 2025 - df['yr_built']

new_houses = df[df['age'] <=50]['price'].values
old_houses = df[df['age'] > 50]['price'].values


bins =  np.linspace(df['price'].min(), df['price'].max(), 11)

new_bins = pandas.cut(new_houses, bins=bins)
old_bins = pandas.cut(old_houses, bins=bins)

new_counts = new_bins.value_counts()
old_counts = old_bins.value_counts()

contingency_table = pandas.DataFrame({
    'New Houses': new_counts,
    'Old Houses': old_counts})

chi2, p_value, dof, ex = chi2_contingency(contingency_table)

print(f"Статистика X^2: {chi2}")
print(f"p-value: {p_value}")
print(f"Степени свободы: {dof}")

if p_value > 0.05:
    print('Выборки можно считать однородными')
else:
    print('Выборки неоднородные')