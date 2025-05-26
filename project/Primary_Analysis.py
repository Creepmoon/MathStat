import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('AppleStock.csv')

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')


df = df[['Date', 'Close']].dropna()

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'])
plt.title('Цена закрытия акций Apple')
plt.xlabel('Дата')
plt.ylabel('Цена ($)')
plt.grid(True)
plt.tight_layout()
plt.show()

df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['LogReturn'])
plt.title('Логарифмическая доходность акций Apple')
plt.xlabel('Дата')
plt.ylabel('Доходность')
plt.grid(True)
plt.tight_layout()
plt.show()
