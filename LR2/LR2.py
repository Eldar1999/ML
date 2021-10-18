import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv('glass.csv')
var_names = list(df.columns)  # получение имен признаков
labels = df.to_numpy('int')[:, -1]  # метки классов
data = df.to_numpy('float')[:, :-1]  # описательные признаки

data = preprocessing.minmax_scale(data)

fig, axs = plt.subplots(2, 4)
fig.set_size_inches(16, 10)
for i in range(data.shape[1] - 1):
    axs[i // 4, i % 4].scatter(data[:, i], data[:, (i + 1)], c=labels)
    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i + 1])
fig.savefig("fig1.png")
plt.close(fig)

pca = PCA(n_components=2)
pca_data = pca.fit(data).transform(data)

print(pca.explained_variance_ratio_)
print(int(sum(pca.explained_variance_ratio_) * 100))
print(pca.singular_values_)

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels)
plt.savefig("fig2.png")
plt.close()

pca = PCA(n_components=4)
pca_data = pca.fit(data).transform(data)

print(pca.explained_variance_ratio_)
print(int(sum(pca.explained_variance_ratio_) * 100))
print(pca.singular_values_)

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels)
plt.savefig("fig3.png")
plt.close()
