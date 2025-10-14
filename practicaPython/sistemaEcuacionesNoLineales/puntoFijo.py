import numpy as np
import pandas as pd
from math import sin, cos, sqrt, exp

# ============================================================
# MÉTODO DE PUNTO FIJO MULTIVARIABLE
# ============================================================


def punto_fijo_multivariable(g, x0, tol=1e-8, maxiter=500):
    x = x0.astype(float).copy()
    history = []
    for k in range(1, maxiter + 1):
        x_new = np.asarray(g(x))  # Calcula g(x_k)
        err = np.linalg.norm(x_new - x, ord=np.inf)  # Norma infinito del cambio
        history.append((k, x.copy(), x_new.copy(), err))
        x = x_new
        if err < tol:  # Condición de parada
            break
    return x, err, k, history


def mostrar_historial_puntofijo(history, names=None):
    rows = []
    for k, x_old, x_new, err in history:
        row = {"iter": k}
        if names is None:
            for i, val in enumerate(x_new):
                row[f"x{i+1}"] = val
        else:
            for i, name in enumerate(names):
                row[name] = x_new[i]
        row["||x_{k+1}-x_k||_inf"] = err
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# EJEMPLO 1: SISTEMA DE 2 VARIABLES
# ============================================================


# Sistema:
# x = cos(y)
# y = sin(x)
def g1(v):
    x, y = v
    return np.array([cos(y), sin(x)])


x0 = np.array([0.5, 0.5])  # Vector inicial

sol, err, iters, hist = punto_fijo_multivariable(g1, x0, tol=1e-10, maxiter=500)
df = mostrar_historial_puntofijo(hist, names=["x", "y"])

print("===== MÉTODO DE PUNTO FIJO MULTIVARIABLE =====")
print("Sistema: x = cos(y), y = sin(x)")
print(f"Solución aproximada: {sol}")
print(f"Iteraciones: {iters}")
print(f"Error final: {err}\n")

print("Tabla de iteraciones:")
print(df.tail())  # Muestra las últimas filas

# ============================================================
# EJEMPLO 2 (opcional): SISTEMA DE 3 VARIABLES
# ============================================================


# x = (cos(y*z)+1)/3
# y = (sin(x)+1)/4
# z = (x*y+1)/5
def g2(v):
    x, y, z = v
    return np.array([(cos(y * z) + 1) / 3, (sin(x) + 1) / 4, (x * y + 1) / 5])


x0_3 = np.array([0.2, 0.2, 0.2])
sol2, err2, iters2, hist2 = punto_fijo_multivariable(g2, x0_3, tol=1e-10, maxiter=500)
df2 = mostrar_historial_puntofijo(hist2, names=["x", "y", "z"])

print("\n===== EJEMPLO DE 3 VARIABLES =====")
print("Sistema: x = (cos(yz)+1)/3, y = (sin(x)+1)/4, z = (xy+1)/5")
print(f"Solución aproximada: {sol2}")
print(f"Iteraciones: {iters2}")
print(f"Error final: {err2}\n")
print(df2.tail())
