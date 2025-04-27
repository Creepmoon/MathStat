import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats


## guess for price distribution

df = pandas.read_csv('kc_house_data.csv')

prices = df['price']
print(f"mean: {prices.mean()}")
print(f"median: {prices.median()}")
print(f"mode:{prices.mode()}")


sns.histplot(prices, kde=True)
plt.title("Распределение цен на жилье")
plt.show()