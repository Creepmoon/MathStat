import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('AppleStock.csv')
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

df['LogReturn'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
df['LogReturn_lag'] = df['LogReturn'].shift(1)
df = df.dropna(subset=['LogReturn', 'LogReturn_lag'])


X = df['LogReturn_lag']
y = df['LogReturn']

X_with_const = sm.add_constant(X)

model = sm.OLS(y, X_with_const).fit()

print(model.summary())
print(f"\n p-value : {model.pvalues['LogReturn_lag']}")
print(f" R^2 : {model.rsquared}")
print(f" STD ошибка беты : {model.bse['LogReturn_lag']}")
print(f" Оценка беты : {model.params['LogReturn_lag']}")



alpha = model.params['const']
beta = model.params['LogReturn_lag']

plt.figure(figsize=(8, 5))

plt.plot(X, alpha + beta * X, color='red', label=f'Линия регрессии:\n$r_i = {alpha:.4f} + {beta:.4f} \cdot r_{{t-1}}$')
plt.title('Линейная регрессия: $r_i$ от $r_{i-1}$')
plt.xlabel('$r_{i-1}$')
plt.ylabel('$r_i$')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
