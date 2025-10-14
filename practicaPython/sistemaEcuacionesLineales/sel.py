import numpy as np


def jacobi(A, b, tol=1e-6, maxiter=100):
    n = len(b)
    x = np.zeros(n)
    for k in range(maxiter):
        x_new = np.zeros(n)
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new
        x = x_new
    return x


A = np.array([[10, 1], [1, 10]])

b = np.array([9, 20])


print("Solucion: ", jacobi(A, b))
