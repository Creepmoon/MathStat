import numpy as np
import pandas as pd
from scipy import stats


file = pd.read_csv('cars93.csv')
df = file[['Price', 'MPG.city', 'MPG.highway', 'Horsepower']].dropna()

X = df[['MPG.city', 'MPG.highway', 'Horsepower']].values
y = df['Price'].values
n = len(y)


X_with_intercept = np.hstack((np.ones((n, 1)), X))
k = X_with_intercept.shape[1]


XtX = X_with_intercept.T @ X_with_intercept
XtX_inv = np.linalg.inv(XtX)
XtY = X_with_intercept.T @ y
beta_hat = XtX_inv @ XtY


y_hat = X_with_intercept @ beta_hat
residuals = y - y_hat


sigma2_hat = np.sum(residuals ** 2) / (n - k)


var_beta_hat = sigma2_hat * XtX_inv
se_beta_hat = np.sqrt(np.diag(var_beta_hat))
t_stats = beta_hat / se_beta_hat
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha / 2, df=n - k)

conf_ints = np.array([
    beta_hat - t_crit * se_beta_hat,
    beta_hat + t_crit * se_beta_hat
]).T

conf_df = pd.DataFrame(conf_ints, columns=["Нижняя граница", "Верхняя граница"],
                       index=["intercept", "MPG.city", "MPG.highway", "Horsepower"])


ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum(residuals ** 2)
r_squared = 1 - ss_res / ss_total


t_stat_hp = t_stats[3]
p_value_hp_one_sided = 1 - stats.t.cdf(t_stat_hp, df=n - k)


t_stat_city = t_stats[1]
p_value_city_two_sided = 2 * (1 - stats.t.cdf(abs(t_stat_city), df=n - k))


X_reduced = df[['Horsepower']].values
X_reduced_with_intercept = np.hstack((np.ones((X_reduced.shape[0], 1)), X_reduced))
beta_reduced = np.linalg.inv(X_reduced_with_intercept.T @ X_reduced_with_intercept) @ \
               (X_reduced_with_intercept.T @ y)
y_hat_reduced = X_reduced_with_intercept @ beta_reduced
rss_reduced = np.sum((y - y_hat_reduced) ** 2)
rss_full = np.sum(residuals ** 2)

q = 2
f_stat = ((rss_reduced - rss_full) / q) / (rss_full / (n - k))
f_p_value = 1 - stats.f.cdf(f_stat, dfn=q, dfd=n - k)


print("\n", "-"*15, "ОЦЕНКИ КОЭФФИЦИЕНТОВ", "-"*15)
for name, b, se, t in zip(["intercept", "MPG.city", "MPG.highway", "Horsepower"],
                          beta_hat, se_beta_hat, t_stats):
    print(f"{name:12s}: β̂ = {b:.4f}, SE = {se:.4f}, t = {t:.4f}")

print("\n", "-"*15, "ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ (95%)", "-"*15)
print(conf_df)

print("\n", "-"*15, "КАЧЕСТВО МОДЕЛИ", "-"*15)
print(f"Остаточная дисперсия: {sigma2_hat:.4f}")
print(f"R²: {r_squared:.4f}")

print("\n", "-"*15,  "ПРОВЕРКА ГИПОТЕЗ", "-"*15)
print(f"(a) t-статистика для Мощности: {t_stat_hp:.4f}, p-value (односторонний): {p_value_hp_one_sided:.5f}")
print(f"(b) t-статистика для mpg.city:   {t_stat_city:.4f}, p-value (двусторонний): {p_value_city_two_sided:.5f}")
print(f"(c) F-статистика для H₀: β_city = β_highway = 0: {f_stat:.4f}, p-value: {f_p_value:.5f}")
