import pandas
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

df = pandas.read_csv('kc_house_data.csv')

df['age'] = 2025 - df['yr_built']

new_houses = df[df['age'] <= 50]['price'].values
old_houses = df[df['age'] > 50]['price'].values

all_values = np.unique(np.concatenate((new_houses, old_houses)))

F_new = np.searchsorted(new_houses, all_values, side='right') / len(new_houses)
F_old = np.searchsorted(old_houses, all_values, side='right') / len(old_houses)

D_stat = np.max(np.abs(F_new - F_old))

n1 = len(new_houses)
n2 = len(old_houses)
ne = n1 * n2 / (n1 + n2)

lambda_val = (np.sqrt(ne) + 0.12 + 0.11/np.sqrt(ne)) * D_stat
p_value_smirnov = 2 * np.exp(-2 * lambda_val ** 2)

if p_value_smirnov > 0.05:
    print("Выборки можно считать однородными")
else:
    print("Выборки неоднородны")
print(f"Статистика смирнова D: {D_stat}, \n p-value: {p_value_smirnov}")

idx = np.argmax(np.abs(F_new - F_old))
plt.figure(figsize=(10,10))
plt.step(all_values, F_new, where='post', label='Новые дома', color='red')
plt.step(all_values, F_old, label='Старые дома', color='blue')
plt.vlines(all_values[idx], F_new[idx], F_old[idx], color = 'black' ,linestyles='-.',label=f'Max_D = {D_stat}')
plt.title('Сравнение двух эмприрических функций')
plt.xlabel('Цена дома')
plt.ylabel('F(x)')
plt.legend()
plt.grid(True)
plt.show()
