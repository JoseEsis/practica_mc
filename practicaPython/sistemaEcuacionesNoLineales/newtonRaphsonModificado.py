import numpy as np
import pandas as pd
from math import sin, cos, exp, sqrt

# ==============================================================
# MÉTODO NEWTON–RAPHSON MODIFICADO
# ==============================================================


def newton_modificado(f, J, x0, tol=1e-8, maxiter=500):

    x = x0.astype(float).copy()
    J0 = np.asarray(J(x0))  # Jacobiano fijo
    history = []

    try:
        for k in range(1, maxiter + 1):
            fx = np.asarray(f(x))
            delta = np.linalg.solve(J0, -fx)
            x_new = x + delta
            err = np.linalg.norm(delta, ord=np.inf)
            resnorm = np.linalg.norm(fx, ord=2)
            history.append((k, x.copy(), delta.copy(), err, resnorm))
            x = x_new
            if err < tol and resnorm < tol:
                break
        return x, err, k, history
    except Exception as e:
        return None, str(e), 0, history


def mostrar_historial_newton(history, names=None):
    """
    Convierte el historial del método en una tabla (DataFrame)
    """
    rows = []
    for k, x_old, delta, err, resnorm in history:
        row = {"iter": k}
        if names is None:
            for i, val in enumerate(x_old):
                row[f"x{i+1}"] = val
        else:
            for i, name in enumerate(names):
                row[name] = x_old[i]
        for i, d in enumerate(delta):
            row[f"delta{i+1}"] = d
        row["||delta||_inf"] = err
        row["||f(x)||_2"] = resnorm
        rows.append(row)
    return pd.DataFrame(rows)


# ==============================================================
# 📘 EJEMPLOS DE USO
# ==============================================================

# Ejemplo 1: Sistema de 2 ecuaciones
# f1 = x^2 + y^2 - 4
# f2 = exp(x) + y - 1
print("EJEMPLO 1 – Sistema de 2 variables")


def f1(v):
    x, y = v
    return np.array([x**2 + y**2 - 4, np.exp(x) + y - 1])


def J1(v):
    x, y = v
    return np.array([[2 * x, 2 * y], [np.exp(x), 1]])


x0_1 = np.array([0.0, 2.0])

sol1, err1, iters1, hist1 = newton_modificado(f1, J1, x0_1, tol=1e-10, maxiter=50)
df1 = mostrar_historial_newton(hist1, names=["x", "y"])

print(f"Solución: {sol1}")
print(f"Iteraciones: {iters1}, Error final: {err1}\n")
print(df1.tail(), "\n")


# Ejemplo 2: Sistema de 3 ecuaciones
# f1 = x + y + z - 3
# f2 = x^2 + y^2 + z^2 - 5
# f3 = exp(x) + y - z - 1
print("EJEMPLO 2 – Sistema de 3 variables")


def f2(v):
    x, y, z = v
    return np.array([x + y + z - 3, x**2 + y**2 + z**2 - 5, np.exp(x) + y - z - 1])


def J2(v):
    x, y, z = v
    return np.array([[1, 1, 1], [2 * x, 2 * y, 2 * z], [np.exp(x), 1, -1]])


x0_2 = np.array([0.0, 2.0, 1.0])

sol2, err2, iters2, hist2 = newton_modificado(f2, J2, x0_2, tol=1e-10, maxiter=50)
df2 = mostrar_historial_newton(hist2, names=["x", "y", "z"])

print(f"Solución: {sol2}")
print(f"Iteraciones: {iters2}, Error final: {err2}\n")
print(df2.tail(), "\n")
