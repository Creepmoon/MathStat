import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('../sex_bmi_smokers.csv')
count_smoking_men = df[(df['sex'] == 'male') & (df['smoker'] == 'yes')].shape[0]
count_nonsmoking_women = df[(df['sex'] == 'female') & (df['smoker'] == 'no')].shape[0]

print(f'Курящих мужчин: {count_smoking_men}')
print(f'Некурящих женщин: {count_nonsmoking_women}')


bmi_all = df['bmi']
print('\n[Все наблюдения]')
print('Среднее:', bmi_all.mean())
print('Дисперсия:', bmi_all.var(ddof=1))
print('Медиана:', bmi_all.median())
print('Квантиль 3/5:', bmi_all.quantile(0.6))


grouped = df.groupby(['sex', 'smoker'])

for name, group in grouped:
    bmi = group['bmi']
    print(f'\nГруппа {name}:')
    print('  Среднее:', bmi.mean())
    print('  Дисперсия:', bmi.var(ddof=1))
    print('  Медиана:', bmi.median())
    print('  Квантиль 3/5:', bmi.quantile(0.6))

def plot_ecdf(data, label=None):
    x = np.sort(data)
    y = np.arange(1, len(x)+1) / len(x)
    plt.step(x, y, where='post', label=label)

plt.figure(figsize=(10, 6))
plot_ecdf(bmi_all, 'Все')
for name, group in grouped:
    plot_ecdf(group['bmi'], f'{name[0]}-{name[1]}')
plt.title('ECDF: Индекс массы тела')
plt.xlabel('BMI')
plt.ylabel('Эмпирическая вероятность')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(bmi_all, bins=20, alpha=0.5, label='Все')
for name, group in grouped:
    plt.hist(group['bmi'], bins=20, alpha=0.5, label=f'{name[0]}-{name[1]}')
plt.title('Гистограмма ИМТ')
plt.xlabel('BMI')
plt.ylabel('Частота')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
df['group'] = df['sex'] + '-' + df['smoker']
df.boxplot(column='bmi', by='group')
plt.title('Boxplot ИМТ по группам')
plt.suptitle('')
plt.xlabel('Группа')
plt.ylabel('BMI')
plt.grid()
plt.show()
