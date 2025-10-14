import numpy as np


def gauss_eliminacion(A, b):
    n = len(b)
    M = np.hstack([A.astype(float), b.reshape(-1, 1)])

    for k in range(n):
        max_row = np.argmax(M[k:, k]) + k
        M[[k, max_row]] = M[[max_row, k]]

        for i in range(k + 1, n):
            factor = M[i, k] / M[k, k]
            M[i] = M[i] - factor * M[k]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i + 1 : n], x[i + 1 : n])) / M[i, i]
    return x


A = np.array([[2, 3], [1, -1]])

b = np.array([[8, 1]])

print("Solucion Ejercicio:\n", gauss_eliminacion(A, b))
