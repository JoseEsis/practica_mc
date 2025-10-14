import numpy as np


def gaus_jordan(A, b):
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1)])

    for k in range(n):
        M[k] = M[k] / M[k, k]

        for i in range(n):
            if i != k:
                M[i] = M[i] - M[i, k] * M[k]
    return M[:, -1]


A = np.array([[3, 2], [1, -2]])
b = np.array([30, -4])


print("Solucion Ejercicio:\n", gaus_jordan(A, b))
