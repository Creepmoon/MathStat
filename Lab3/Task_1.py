import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as sp

def kolmogorov_smirnov_p_value(D, n, terms=100):

    lambda_val = (np.sqrt(n) + 0.12 + 0.11 / np.sqrt(n)) * D
    sum_series = 0
    for j in range(1, terms):
        sum_series += (-1)**(j-1) * np.exp(-2 * (lambda_val**2) * (j**2))
    p_value = 2 * sum_series
    return max(min(p_value, 1), 0)



df = pandas.read_csv('kc_house_data.csv')

prices = df['price'].dropna().values

scale_prices = np.sort((prices - prices.mean()) / prices.std())

length = len(scale_prices)

F_empir = np.arange(1, length + 1) / length
F_theory = sp.norm.cdf(scale_prices)

D1 = F_empir - F_theory
D2 = F_theory - np.concatenate(([0], F_empir[:-1]))

D_stat = max(np.max(D1),np.max(D2))

D_crit =  1.35810/np.sqrt(length)

p_value_manual = kolmogorov_smirnov_p_value(D_stat, length)


max_diff_idx = np.argmax(np.abs(F_empir - F_theory))
x_max = scale_prices[max_diff_idx]
y_empirical_max = F_empir[max_diff_idx]
y_theoretical_max = F_theory[max_diff_idx]

plt.figure(figsize=(10, 6))
plt.step(scale_prices, F_empir, where='post', label='Эмпирическая CDF', color='red')
plt.plot(scale_prices, F_theory, label='Теоретическая (нормальная) CDF', color='blue')
plt.vlines(x=x_max, ymin=y_theoretical_max, ymax=y_empirical_max, color='black', linestyle='-.', label=f'D = {D_stat:.4f}')
plt.title('Сравнение эмпирической и теоретической функции распределения')
plt.xlabel('Стандартизированная цена')
plt.ylabel('F(x)')
plt.legend()
plt.grid()
plt.show()

print(f"\n Cтатистика: {D_stat}, \n Критическое значение (alpha = 0.05): {D_crit}")
print(f" p_value: {p_value_manual}")

if D_stat > D_crit:
    print("Гипотеза о норм распределении отвергается")
else:
    print("Гипотеза о норм распределении подтверждается")

