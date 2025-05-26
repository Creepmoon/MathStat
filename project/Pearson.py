import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


df = pd.read_csv('AppleStock.csv')
df.columns = df.columns.str.strip()

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

df['LogReturn'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))

df = df.dropna(subset=['LogReturn']).copy()

df['LogReturn_lag'] = df['LogReturn'].shift(1)
df = df.dropna(subset=['LogReturn_lag'])

corr, p_value_corr = pearsonr(df['LogReturn'], df['LogReturn_lag'])

print(f"Коэффициент корреляции Пирсона : {corr:.6f}")
print(f"p-value: {p_value_corr:.6f}")


plt.figure(figsize=(8, 5))
plt.scatter(df['LogReturn_lag'], df['LogReturn'], alpha=0.5)
plt.title('Диаграмма рассеяния: $r_{i}$ от $r_{i-1}$ (корреляция Пирсона)')
plt.xlabel('$r_{i-1}$')
plt.ylabel('$r_i$')
plt.grid(True)
plt.tight_layout()
plt.show()