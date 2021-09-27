import numpy as np
from matplotlib import pyplot as plt


def k1(x1, x2):
    return (x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2
    pass


# Part 1
print("Part 1")
X = np.array([[4, 2.9], [2.5, 1], [3.5, 4], [2, 2.1]])
print(X.tolist())
for x in X:
    plt.plot(x[0], x[1], "o")
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.grid()
plt.savefig("test.png")
plt.close()

print("Kernel matrix")
for i in range(4):
    for j in range(4):
        print("%2.4f" % k1(X[i], X[j]), end=" ")
    print()

# part 2
print("Part 2")

D = np.array([[8, -20], [0, -1], [10, -19], [10, -20], [2, 0]])
print(D)
print("Mean: ", end=" ")
print(np.mean(D, 0))
print("Cov matrix")
cov_matrix = np.cov(np.array([x[0] for x in D]), np.array([x[1] for x in D]))
print(cov_matrix)
for x in D:
    plt.plot(x[0], x[1], "o")
plt.xlim(-12, 12)
plt.grid()
plt.savefig("test2.png")
plt.close()

print("Eigenvalues", end=" ")
print(str(np.linalg.eigvals(cov_matrix)))

print("Corr matrix")
print(np.corrcoef(np.array([x[0] for x in D]), np.array([x[1] for x in D])))

# part 3
print("Part 3")
X = D - np.mean(D, 0)
print(X)
for x in X:
    plt.plot(x[0], x[1], "o")
plt.xlim(-12, 12)
plt.grid()
plt.savefig("test3.png")
plt.close()

eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
print("Main components", end=" ")
print(eigen_vectors)

print("Residual variance", end=" ")
print(eigen_values)

print("First component projection", end=" ")
print(np.dot(X, eigen_vectors[:, 0]))

print("Second component projection", end=" ")
print(np.dot(X, eigen_vectors[:, 1]))