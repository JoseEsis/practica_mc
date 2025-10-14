import numpy as np
import math


def f(x):
    return x**3 - x - 2


def biseccion(f, a, b, tol=1e-8, maxit=100):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("No hay cambio de signo en [a,b]")
    rows = []
    for k in range(1, maxit + 1):
        r = (a + b) / 2.0
        fr = f(r)
        rows.append((k, a, b, r, fr))
        if abs(fr) < tol or (b - a)/2.0 < tol:
            break
        if fa * fr < 0:
            b, fb = r, fr
        else:
            a, fa = r, fr
    return rows


a, b = 1.0, 2.0
bis = biseccion(f, a, b)


for row in bis:
    print(row)

print("biseccion = ", bis[-1][3], "iter", bis[-1][0])
