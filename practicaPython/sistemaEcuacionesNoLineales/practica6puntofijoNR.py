import numpy as np
import pandas as pd
from math import sin, cos, sqrt, exp

# ==============================================================
# FUNCIONES GENERALES
# ==============================================================

def punto_fijo_multivariable(g, x0, tol=1e-8, maxiter=500):
    x = x0.astype(float).copy()
    history = []
    for k in range(1, maxiter+1):
        x_new = np.asarray(g(x))
        err = np.linalg.norm(x_new - x, ord=np.inf)
        history.append((k, x.copy(), x_new.copy(), err))
        x = x_new
        if err < tol:
            break
    return x, err, k, history

def newton_modificado(f, J, x0, tol=1e-8, maxiter=500):
    x = x0.astype(float).copy()
    J0 = np.asarray(J(x0))
    history = []
    try:
        for k in range(1, maxiter+1):
            fx = np.asarray(f(x))
            delta = np.linalg.solve(J0, -fx)
            x_new = x + delta
            err = np.linalg.norm(delta, ord=np.inf)
            resnorm = np.linalg.norm(fx, ord=2)
            history.append((k, x.copy(), delta.copy(), err))
            x = x_new
            if err < tol and resnorm < tol:
                break
        return x, err, k, history
    except Exception as e:
        return  None, str(e), 0, history

def mostrar_historial_puntofijo(history, names=None):
    rows = []
    for k, x_old, x_new, err in history:
        row = {"iter": k}
        if names is None:
            for i, val in enumerate(x_new):
                row[f"x{i}"] = val
        else:
            for i, name in enumerate(names):
                row[name] = x_new[i]
        row["||x_{k+1}-x_k||_inf"] = err
        rows.append(row)
    return pd.DataFrame(rows)

def mostrar_historial_newton(history, names=None):
    rows = []
    for k, x_old, delta, err, resnorm in history:
        row = {"iter": k}
        if names is None:
            for i, val in enumerate(x_old):
                row[f"x{i}"] = val
        else:
            for i, name in enumerate(names):
                row[name] = x_old[i]
        for i, d in enumerate(delta):
            row[f"delta{i}"] = d
        row["||delta||_inf"] = err
        row["||f(x)||_2"] = resnorm
        rows.append(row)
    return pd.DataFrame(rows)

# ==============================================================
# EJERCICIOS DE PUNTO FIJO
# ==============================================================

print('Ejercicio PF1: x = cos(y), y = sin(x)')
def g1(v):
    x, y = v
    return np.array([cos(y), sin(x)])
x0_1 = np.array([0.5, 0.5])

print('Ejercicio PF2: x = sqrt((1+y)/2), y = sqrt((1+x)/2)')
def g2(v):
    x, y = v
    return np.array([sqrt((1+y)/2), sqrt((1+x)/2)])
x0_2 = np.array([0.5, 0.5])

print('Ejercicio PF3: x = (sin(y)+1)/2, y = (cos(x)+1)/2')
def g3(v):
    x, y = v
    return np.array([(sin(y)+1)/2, (cos(x)+1)/2])
x0_3 = np.array([0.3, 0.3])

print('Ejercicio PF4: sistema 3 variables ')
def g4(v):
    x, y, z = v
    return np.array([(cos(y*z)+1)/2, (sin(x)+1)/3, (x*y + 1)/4])
x0_4 = np.array([0.2, 0.2, 0.2])

print('x = (cos(yz)+1)/3, y = (sin(x)+1)/4, z = (xy+1)/5')
print('Ejercicio PF5: Ejercicio 9 ')
def g5(v):
    x, y, z = v
    return np.array([(cos(y*z)+1)/3, (sin(x)+1)/4, (x*y+1)/5])
x0_5 = np.array([0.2, 0.2, 0.2])

# 🔹 Ejercicio 10 – Punto Fijo de 3 variables
print('x = sqrt((1+y+z)/4), y = sqrt((1+x+z)/4), z = sqrt((1+x+y)/4)')
print('Ejercicio PF6: Ejercicio 10 ')
def g6(v):
    x, y, z = v
    return np.array([sqrt((1+y+z)/4), sqrt((1+x+z)/4), sqrt((1+x+y)/4)])
x0_6 = np.array([0.5, 0.5, 0.5])

# ==============================================================
# EJERCICIOS NEWTON–RAPHSON MODIFICADO
# ==============================================================

print('Ejercicio NRM1: f1 = x^2+y^2-4, f2 = exp(x)+y-1')
def f_nr1(v):
    x, y = v
    return np.array([x**2 + y**2 - 4, np.exp(x) + y - 1])
def J_nr1(v):
    x, y = v
    return np.array([[2*x, 2*y], [np.exp(x), 1.0]])
x0_nr1 = np.array([0.0, 2.0])

print(' Ejercicio NRM2: sistema 3 variables')
def f_nr2(v):
    x, y, z = v
    return np.array([x + y + z - 3, x**2 + y**2 + z**2 - 5, np.exp(x) + y - z - 1])
def J_nr2(v):
    x, y, z = v
    return np.array([[1, 1, 1], [2*x, 2*y, 2*z], [np.exp(x), 1, -1]])
x0_nr2 = np.array([0.0, 2.0, 1.0])

print('Ejercicio NRM3: f1 = x^2 - y -1, f2 = y^2 - x -1')
def f_nr3(v):
    x, y = v
    return np.array([x**2 - y - 1, y**2 - x - 1])
def J_nr3(v):
    x, y = v
    return np.array([[2*x, -1], [-1, 2*y]])
x0_nr3 = np.array([1.5, 1.5])

print('Ejercicio NRM4: f1 = sin(x)+y-1, f2 = x+cos(y)-0.5')
def f_nr4(v):
    x, y = v
    return np.array([sin(x) + y - 1, x + cos(y) - 0.5])
def J_nr4(v):
    x, y = v
    return np.array([[cos(x), 1], [1, -sin(y)]])
x0_nr4 = np.array([0.5, 0.5])

# ==============================================================
# MAIN
# ==============================================================

if __name__ == "__main__":

    print("\n===== PUNTO FIJO =====")
    for i, (g, x0, name) in enumerate([
        (g1, x0_1, "PF1: cos/sen"),
        (g2, x0_2, "PF2: sqrt-symmetric"),
        (g3, x0_3, "PF3: sen/cos scaled"),
        (g4, x0_4, "PF4: 3-variable ejemplo"),
        (g5, x0_5, "PF5: Ejercicio 9"),
        (g6, x0_6, "PF6: Ejercicio 10")
    ]):
        sol, err, iters, hist = punto_fijo_multivariable(g, x0, tol=1e-10, maxiter=500)
        df = mostrar_historial_puntofijo(hist, names=[f"x{i+1}" for i in range(len(x0))])
        print(f"{name}: solución = {sol}, iteraciones = {iters}, error final = {err}")
        if not df.empty:
            print(df.tail(), "\n")
        else:
            print("No hay iteraciones registradas.\n")

    print("\n===== NEWTON–RAPHSON MODIFICADO =====")
    for i, (f, J, x0, name) in enumerate([
        (f_nr1, J_nr1, x0_nr1, "NRM1: circle+exp"),
        (f_nr2, J_nr2, x0_nr2, "NRM2: 3-var"),
        (f_nr3, J_nr3, x0_nr3, "NRM3: symmetric quad"),
        (f_nr4, J_nr4, x0_nr4, "NRM4: sin/cos mix")
    ]):
        sol, err, iters, hist = newton_modificado(f, J, x0, tol=1e-10, maxiter=50)
        if sol is None:
            print(f"{name}: error -> {err}")
        else:
            df = mostrar_historial_newton(hist, names=[f"x{i+1}" for i in range(len(x0))])
            print(f"{name}: solución = {sol}, iteraciones = {iters}, error final = {err}")
            if not df.empty:
                print(df.tail(), "\n")
            else:
                print("No hay iteraciones registradas.\n")