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

print([["%2.5e " % numpy.mean(x), "%2.5e" % numpy.std(x)] for x in data])
print([["%2.5e " % numpy.mean(x), "%2.5e" % numpy.std(x)] for x in data_scaled])
