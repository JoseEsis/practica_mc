import numpy as np
import pandas as pd
from math import cos, sin


def puntoFijo(g, x0, tol=1e-8, maxiter=500):
    x = x0.astype(float).copy()
    history = []
    for k in range(1, maxiter + 1):
        x_new = np.asarray(g(x))
        err = np.linalg.norm(x_new - x, ord=np.inf)
        history.append((k, x.copy(), x_new.copy(), err))
        x = x_new
        if err < tol:
            break
    return x, k, err, history


def mostrar(history, names=None):
    rows = []
    for k, x_old, x_new, err in history:
        row = {"iter": k}
        if names is None:
            for i, val in enumerate(x_new):
                row[f"x{i+1}"] = val
        else:
            for i, name in enumerate(names):
                row[name] = x_new[i]
        row["err"] = err
        rows.append(row)
    return pd.DataFrame(rows)


def g1(v):
    x, y = v
    return np.array([cos(y), sin(x)])


x0 = np.array([0.5, 0.5])

sol1, err, iters, hist = puntoFijo(g1, x0, tol=1e-10, maxiter=500)
d = mostrar(hist, names=["x", "y"])
print(f"sol: {sol1}")

print(d.head())
