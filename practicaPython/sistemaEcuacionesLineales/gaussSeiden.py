import numpy as np


def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s = sum(A[i, j] * x_new[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            print(f"Convergio {k+1} iteraciones")
            return x_new
        x = x_new
    return x


A = np.array([[10, 1], [1, 10]])

b = np.array([9, 20])

print("Solución Gasuss_Seidel: ", gauss_seidel(A, b))
