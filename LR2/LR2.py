import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA

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

# PCA with 2 components

pca = PCA(n_components=2)
pca_data = pca.fit(data).transform(data)

print(pca.explained_variance_ratio_)
print(int(sum(pca.explained_variance_ratio_) * 100))
print(pca.singular_values_)

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels)
plt.savefig("fig2.png")
plt.close()

# PCA with 4 components

pca = PCA(n_components=4)
pca_data = pca.fit(data).transform(data)

print(pca.explained_variance_ratio_)
print(int(sum(pca.explained_variance_ratio_) * 100))
print(pca.singular_values_)

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels)
plt.savefig("fig3.png")
plt.close()

pca_data = pca.inverse_transform(pca_data)

fig, axs = plt.subplots(2, 4)
fig.set_size_inches(16, 10)
for i in range(data.shape[1] - 1):
    axs[i // 4, i % 4].scatter(pca_data[:, i], pca_data[:, (i + 1)], c=labels)
    axs[i // 4, i % 4].set_xlabel(var_names[i])
    axs[i // 4, i % 4].set_ylabel(var_names[i + 1])
fig.savefig("fig4.png")
plt.close(fig)

# PCA with 2 components with svd_solver args

pca = PCA(n_components=2, svd_solver="full")
pca_data = pca.fit(data).transform(data)

print(pca.explained_variance_ratio_)
print(int(sum(pca.explained_variance_ratio_) * 100))
print(pca.singular_values_)

plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels)
plt.savefig("fig5.png")
plt.close()

# KernelPCA

kernel_pca = KernelPCA(n_components=2)
kernel_pca_data = kernel_pca.fit_transform(data)

print(kernel_pca)

plt.scatter(kernel_pca_data[:, 0], kernel_pca_data[:, 1], c=labels)
plt.savefig("fig6.png")
plt.close()