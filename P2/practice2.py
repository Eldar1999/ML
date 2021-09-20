import numpy as np
from matplotlib import pyplot as plt


def k1(x1, x2):
    return np.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
    pass


X = np.array([[4, 2.9], [2.5, 1], [3.5, 4], [2, 2.1]])
print(X.tolist())
for x in X:
    plt.plot(x[0], x[1], "o")
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.show()

for i in range(4):
    for j in range(4):
        print("%2.4f" % k1(X[i], X[j]), end=" ")
    print()

