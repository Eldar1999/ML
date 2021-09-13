import numpy
from scipy import stats
import matplotlib.pyplot as plt


def normal_dist(x, mean, var):
    prob_density = 1 / numpy.sqrt(2 * numpy.pi * var) * numpy.exp(-0.5 * (x - mean) ** 2 / var)
    return prob_density


def cov(X, Y):
    return numpy.mean((X - numpy.mean(X)) * (Y - numpy.mean(Y)))


X = numpy.array([69, 74, 68, 70, 72, 67, 66, 70, 76, 68, 72, 79, 74, 67, 66, 71, 74, 75, 75, 76], numpy.float64)
Y = numpy.array([153, 175, 155, 135, 172, 150, 115, 137, 200, 130, 140, 265, 185, 112, 140, 150, 165, 185, 210, 220],
                numpy.float64)
print(X)
print("Mean ", end="")
print(numpy.average(X))
print("Med ", end="")
print(numpy.median(X))
print("Mode ", end="")
print(stats.mode(X)[0][0])
print("Variance ", end="")
print(numpy.var(Y))
plt.plot(numpy.arange(50, 90, 0.1),
         normal_dist(numpy.array(numpy.arange(50, 90, 0.1)), numpy.mean(X), numpy.var(X)))
plt.grid()
plt.title("Нормальное распределение Х")
plt.savefig("test.png")
plt.close()
print("P(>80) = ", end="")
print(numpy.sum(0.1 * normal_dist(numpy.array(numpy.arange(80.1, 1000, 0.1)), numpy.mean(X), numpy.var(X))))

print("Mean (X Y) = ", end="")
print("(" + str(numpy.mean(X)) + " " + str(numpy.mean(Y)) + ")")

print("Cov matrix")
print(numpy.array([[cov(X, X), cov(X, Y)], [cov(Y, X), cov(Y, Y)]]))

print("Corr matrix")
print(numpy.corrcoef(X, Y))

plt.title("Scatter plot")
plt.scatter(X, Y)
plt.grid()
plt.savefig("test2.png")
plt.close()

print("\nPart 2")
X1 = numpy.array([17, 11, 11])
X2 = numpy.array([17, 9, 8])
X3 = numpy.array([12, 13, 19])

print("Cov matrix")
print(numpy.array([[cov(X1, X1), cov(X1, X2), cov(X1, X3)], [cov(X2, X1), cov(X2, X2), cov(X2, X3)],
                   [cov(X3, X1), cov(X3, X2), cov(X3, X3)]]))

print("Generalized var")
print(numpy.linalg.det(numpy.array([[cov(X1, X1), cov(X1, X2), cov(X1, X3)], [cov(X2, X1), cov(X2, X2), cov(X2, X3)],
                                    [cov(X3, X1), cov(X3, X2), cov(X3, X3)]])))

print("\nPart 3")
p = numpy.array([5, 6, 7])
Na = normal_dist(p, 4, 1)
Nb = normal_dist(p, 8, 4)
plt.plot(p, Na, "rs", label="Na")
plt.plot(p, Nb, "bs", label="Nb")
plt.legend()
plt.grid()
plt.savefig("test3.png")

range1 = numpy.arange(5, 7, 0.01)
tmp1 = normal_dist(range1, 4, 1)
tmp2 = normal_dist(range1, 8, 4)

for i in range(len(range1)):
    if abs(tmp1[i] - tmp2[i]) < 0.0001:
        print("draw ", end="")
        print(range1[i])
        break
