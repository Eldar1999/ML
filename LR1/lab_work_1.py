import numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Part1

df = pd.read_csv('heart_failure_clinical_records_dataset.csv')

df = df.drop(
    columns=['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking', 'time', 'DEATH_EVENT'])

print(df)  # Вывод датафрейма с данными для лаб. работы. Должно быть 299 наблюдений и 6 признаков

n_bins = 20
fig, axs = plt.subplots(2, 3)
fig.set_size_inches(12, 10)
axs[0, 0].hist(df['age'].values, bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(df['creatinine_phosphokinase'].values, bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(df['ejection_fraction'].values, bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(df['platelets'].values, bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(df['serum_creatinine'].values, bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(df['serum_sodium'].values, bins=n_bins)
axs[1, 2].set_title('serum_sodium')
plt.savefig("fig1.png")

data = df.to_numpy(dtype='float')

# Part2

scaler = preprocessing.StandardScaler().fit(data[:150, :])
data_scaled = scaler.transform(data)

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(12, 10)
axs[0, 0].hist(data_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')
plt.savefig("fig2.png")

scaler300 = preprocessing.StandardScaler().fit(data)
data_scaled300 = scaler300.transform(data)

print("Orig")
print("\n\n".join("%2.3e\n" % x + "%2.3e" % y for x, y in zip(numpy.mean(data, 0), numpy.std(data, 0))))
print("Std150")
print("\n\n".join("%2.3e\n" % x + "%2.3e" % y for x, y in zip(numpy.mean(data_scaled, 0), numpy.std(data_scaled, 0))))
print("Std300")
print("\n\n".join(
    "%2.3e\n" % x + "%2.3e" % y for x, y in zip(numpy.mean(data_scaled300, 0), numpy.std(data_scaled300, 0))))

# Part3

min_max_scaler = preprocessing.MinMaxScaler().fit(data)
data_min_max_scaled = min_max_scaler.transform(data)

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(12, 10)
axs[0, 0].hist(data_min_max_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_min_max_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_min_max_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_min_max_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_min_max_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_min_max_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')
plt.savefig("fig3.png")

print("\t".join([str(x) for x in min_max_scaler.data_min_]))
print("\t".join([str(x) for x in min_max_scaler.data_max_]))

data_max_abs_scaled = preprocessing.MaxAbsScaler().fit_transform(data)

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(12, 10)
axs[0, 0].hist(data_max_abs_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_max_abs_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_max_abs_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_max_abs_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_max_abs_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_max_abs_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')
plt.savefig("fig4.png")

data_robust_scaled = preprocessing.RobustScaler().fit_transform(data)

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(12, 10)
axs[0, 0].hist(data_robust_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_robust_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_robust_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_robust_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_robust_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_robust_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')
plt.savefig("fig5.png")

data_min_max_scaled_1 = preprocessing.MinMaxScaler((-5, 10)).fit_transform(data)

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(12, 10)
axs[0, 0].hist(data_min_max_scaled_1[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_min_max_scaled_1[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_min_max_scaled_1[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_min_max_scaled_1[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_min_max_scaled_1[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_min_max_scaled_1[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')
plt.savefig("fig6.png")

# Part4

quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100,
                                                         random_state=0).fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(12, 10)
axs[0, 0].hist(data_quantile_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_quantile_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_quantile_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_quantile_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_quantile_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_quantile_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')
plt.savefig("fig7.png")

quantile_transformer = preprocessing.QuantileTransformer(n_quantiles=100,
                                                         random_state=0, output_distribution='normal').fit(data)
data_quantile_scaled = quantile_transformer.transform(data)

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(12, 10)
axs[0, 0].hist(data_quantile_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_quantile_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_quantile_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_quantile_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_quantile_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_quantile_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')
plt.savefig("fig8.png")

data_power_transformer_quantile_scaled = preprocessing.PowerTransformer().fit_transform(data)

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(12, 10)
axs[0, 0].hist(data_power_transformer_quantile_scaled[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(data_power_transformer_quantile_scaled[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(data_power_transformer_quantile_scaled[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(data_power_transformer_quantile_scaled[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(data_power_transformer_quantile_scaled[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(data_power_transformer_quantile_scaled[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')
plt.savefig("fig9.png")

# Part 5

discretizer = preprocessing.KBinsDiscretizer(n_bins=[3, 4, 3, 10, 2, 4],
                                             encode='ordinal')
discretized_data = discretizer.fit_transform(data)

fig, axs = plt.subplots(2, 3)
fig.set_size_inches(12, 10)
axs[0, 0].hist(discretized_data[:, 0], bins=n_bins)
axs[0, 0].set_title('age')
axs[0, 1].hist(discretized_data[:, 1], bins=n_bins)
axs[0, 1].set_title('creatinine_phosphokinase')
axs[0, 2].hist(discretized_data[:, 2], bins=n_bins)
axs[0, 2].set_title('ejection_fraction')
axs[1, 0].hist(discretized_data[:, 3], bins=n_bins)
axs[1, 0].set_title('platelets')
axs[1, 1].hist(discretized_data[:, 4], bins=n_bins)
axs[1, 1].set_title('serum_creatinine')
axs[1, 2].hist(discretized_data[:, 5], bins=n_bins)
axs[1, 2].set_title('serum_sodium')
plt.savefig("fig10.png")

print('bin_edges_')
print(discretizer.bin_edges_)
