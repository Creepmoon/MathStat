from scipy.stats import spearmanr
import pandas

df = pandas.read_csv('kc_house_data.csv')

x = df['sqft_living']
y = df['price']

# Готовая реализация
rho, p_two_tailed = spearmanr(x, y)

# Односторонний p-value
p_one_tailed = p_two_tailed / 2 if rho > 0 else 1 - p_two_tailed / 2

print(f"Коэффициент Спирмена p = {rho:.4f}")
print(f"Односторонний p-value = {p_one_tailed:.6f}")

if p_one_tailed < 0.05:
    print("Отвергаем H0.")
else:
    print("Нет оснований отвергать H0.")
