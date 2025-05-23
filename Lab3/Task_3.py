import pandas
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

df = pandas.read_csv('kc_house_data.csv')

x = df['sqft_living']
y = df['price']

n = len(df)

r = np.corrcoef(x, y)[0, 1]

t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)

alpha = 0.05
t_critical = stats.t.ppf(1-alpha/2, df= n-2)

print(f'Критическое значение t для alpha=0.05: {t_critical}')
print(f'коэффицент t: {t_stat}')
print(f"коэффицент корреляции Пирсона:{r}")
if t_stat > t_critical:
    print('Есть статистически положительная корреляция.')
else:
    print('есть отрицательная корреляция')

plt.figure(figsize = (10,10))
plt.scatter(x,y, alpha = 0.5)
plt.title('Связь между площадью дома и ценной')
plt.xlabel('Площадь')
plt.ylabel('Цена')
plt.grid(True)
plt.show()