import math


def f(x):
    return x**3 - x - 2


def secante(f, x0, x1, tol=1e-8, maxit=100):
    rows = []
    for k in range(1, maxit + 1):
        fx0, fx1 = f(x0), f(x1)
        if fx1 == fx0:
            raise ZeroDivisionError("Denominador cero ")
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        fx2 = f(x2)
        rows.append((k, x0, x1, x2, fx1, fx2))
        if abs(fx2) < tol or abs(x2 - x1) < tol:
            break
        x0, x1 = x1, x2
    return rows


a, b = 1.0, 2.0
sec = secante(f, a, b)

for rows in sec:
    print(rows)
print("Secante", sec[-1][3], "iter", sec[-1][0])
