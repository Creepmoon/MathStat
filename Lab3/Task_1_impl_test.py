import pandas
from scipy.stats import normaltest

df = pandas.read_csv('kc_house_data.csv')

prices = df['price'].dropna().values

statistics, p_value = normaltest(prices)

print(f"Статистика: {statistics}")
print(f"p-value: {p_value}")

if p_value > 0.05:
    print("Гипотеза нормальности не отвергается")
else:
    print("Гипотеза нормальности отвергается")