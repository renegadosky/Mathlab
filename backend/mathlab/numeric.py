"""Engenharia numérica. Todo resultado é aproximado e traz uma medida de erro/resíduo."""
import math
import numpy as np
import sympy as sp
from .engine import MathError, parse_function

X = sp.Symbol("x")


def _fn(expr):
    e = parse_function(expr)
    f = sp.lambdify(X, e, "math")

    def safe(v):
        try:
            r = float(f(v))
        except (ZeroDivisionError, ValueError, OverflowError):
            raise MathError(f"A função não está definida em x = {v}.")
        if not math.isfinite(r):
            raise MathError(f"A função não é finita em x = {v}.")
        return r
    return e, safe


def _mat(a, name="A"):
    try:
        m = np.array(a, dtype=float)
    except (ValueError, TypeError):
        raise MathError(f"{name} deve ser uma matriz numérica retangular.")
    if m.ndim != 2 or m.size == 0 or not np.all(np.isfinite(m)):
        raise MathError(f"{name} deve ser uma matriz 2D finita e não vazia.")
    return m


def _vec(a, name="dados", min_len=1):
    try:
        v = np.array(a, dtype=float)
    except (ValueError, TypeError):
        raise MathError(f"{name} deve ser uma lista numérica.")
    if v.ndim != 1 or len(v) < min_len or not np.all(np.isfinite(v)):
        raise MathError(f"{name} deve ter ao menos {min_len} valor(es) numérico(s) finito(s).")
    return v


def _out(method, **kw):
    return {"method": method, "exact": False, **kw}


def newton_raphson(expr, x0, tol=1e-12, max_iter=50):
    e, f = _fn(expr)
    _, df = _fn(str(sp.diff(e, X)))
    x = float(x0)
    for k in range(1, max_iter + 1):
        d = df(x)
        if d == 0:
            raise MathError(f"Derivada nula em x = {x}; o método falha.")
        nx = x - f(x) / d
        if abs(nx - x) < tol:
            return _out("newton_raphson", root=nx, iterations=k, residual=abs(f(nx)),
                        converged=True, formula=r"x_{n+1}=x_n-\frac{f(x_n)}{f'(x_n)}")
        x = nx
    raise MathError(f"Não convergiu em {max_iter} iterações.")


def bisection(expr, a, b, tol=1e-12, max_iter=200):
    _, f = _fn(expr)
    a, b = float(a), float(b)
    if a >= b:
        raise MathError("É necessário a < b.")
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise MathError("f(a) e f(b) têm o mesmo sinal: não há garantia de raiz no intervalo.")
    for k in range(1, max_iter + 1):
        m = (a + b) / 2
        fm = f(m)
        if fm == 0 or (b - a) / 2 < tol:
            return _out("bisection", root=m, iterations=k, residual=abs(fm), converged=True,
                        formula=r"m=\frac{a+b}{2}")
        if fa * fm < 0:
            b = m
        else:
            a, fa = m, fm
    raise MathError("Não convergiu.")


def _check_interval(a, b, n):
    if not (float(a) < float(b)):
        raise MathError("É necessário a < b.")
    if int(n) < 1:
        raise MathError("n deve ser ≥ 1.")


def trapezoid(expr, a, b, n=1000):
    _check_interval(a, b, n)
    _, f = _fn(expr)
    a, b, n = float(a), float(b), int(n)
    h = (b - a) / n
    s = (f(a) + f(b)) / 2 + sum(f(a + i * h) for i in range(1, n))
    return _out("trapezoid", value=h * s, n=n, formula=r"\frac{h}{2}\left[f(a)+2\sum f(x_i)+f(b)\right]",
                error_order="O(h^2)")


def simpson(expr, a, b, n=1000):
    _check_interval(a, b, n)
    if int(n) % 2:
        raise MathError("A regra de Simpson exige n par.")
    _, f = _fn(expr)
    a, b, n = float(a), float(b), int(n)
    h = (b - a) / n
    s = f(a) + f(b) + 4 * sum(f(a + i * h) for i in range(1, n, 2)) + 2 * sum(f(a + i * h) for i in range(2, n, 2))
    return _out("simpson", value=h * s / 3, n=n, formula=r"\frac{h}{3}\left[f_0+4f_1+2f_2+\dots+f_n\right]",
                error_order="O(h^4)")


def numeric_derivative(expr, x0, h=1e-5):
    _, f = _fn(expr)
    x0, h = float(x0), float(h)
    if h <= 0:
        raise MathError("h deve ser positivo.")
    return _out("central_difference", value=(f(x0 + h) - f(x0 - h)) / (2 * h),
                formula=r"f'(x)\approx\frac{f(x+h)-f(x-h)}{2h}", error_order="O(h^2)")


def lagrange(xs, ys):
    xs, ys = _vec(xs, "xs", 2), _vec(ys, "ys", 2)
    if len(xs) != len(ys):
        raise MathError("xs e ys devem ter o mesmo tamanho.")
    if len(set(xs.tolist())) != len(xs):
        raise MathError("Os valores de x devem ser distintos.")
    pts = [(sp.nsimplify(a), sp.nsimplify(b)) for a, b in zip(xs.tolist(), ys.tolist())]
    poly = sp.expand(sp.interpolate(pts, X))
    return {"method": "lagrange", "exact": True, "polynomial": str(poly), "latex": sp.latex(poly),
            "degree": sp.degree(poly, X) if poly.has(X) else 0}


def solve_linear(A, b):
    A, b = _mat(A), _vec(b, "b")
    if A.shape[0] != A.shape[1] or A.shape[0] != len(b):
        raise MathError("A deve ser quadrada n×n e b deve ter tamanho n.")
    if np.linalg.matrix_rank(A) < A.shape[0]:
        raise MathError("Sistema singular: sem solução única.")
    x = np.linalg.solve(A, b)
    return _out("solve_linear", solution=x.tolist(), residual=float(np.linalg.norm(A @ x - b)),
                condition_number=float(np.linalg.cond(A)))


def lu(A):
    A = _mat(A)
    n = A.shape[0]
    if n != A.shape[1]:
        raise MathError("A decomposição LU exige matriz quadrada.")
    U, L, P = A.copy(), np.eye(n), np.eye(n)
    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(U[k:, k])))
        if U[p, k] == 0:
            continue
        U[[k, p]], P[[k, p]], L[[k, p], :k] = U[[p, k]], P[[p, k]], L[[p, k], :k]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i] -= L[i, k] * U[k]
    return _out("lu", P=P.tolist(), L=L.tolist(), U=np.triu(U).tolist(),
                residual=float(np.linalg.norm(P @ A - L @ U)), relation="PA = LU")


def qr(A):
    A = _mat(A)
    Q, R = np.linalg.qr(A)
    return _out("qr", Q=Q.tolist(), R=R.tolist(), residual=float(np.linalg.norm(Q @ R - A)),
                orthogonality_error=float(np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))), relation="A = QR")


def _c(z):
    return float(z.real) if abs(z.imag) < 1e-12 else {"re": float(z.real), "im": float(z.imag)}


def eig(A):
    A = _mat(A)
    if A.shape[0] != A.shape[1]:
        raise MathError("Autovalores exigem matriz quadrada.")
    w, v = np.linalg.eig(A)
    res = max(float(np.linalg.norm(A @ v[:, i] - w[i] * v[:, i])) for i in range(len(w)))
    return _out("eig", eigenvalues=[_c(z) for z in w],
                eigenvectors=[[_c(z) for z in v[:, i]] for i in range(len(w))],
                max_residual=res, note="Cada autovetor está em eigenvectors[i], normalizado.")


def describe(data):
    v = _vec(data, "dados", 1)
    out = {"n": len(v), "mean": float(v.mean()), "median": float(np.median(v)),
           "min": float(v.min()), "max": float(v.max())}
    if len(v) >= 2:
        out["variance_sample"] = float(v.var(ddof=1))
        out["std_sample"] = float(v.std(ddof=1))
    out["variance_population"] = float(v.var())
    return _out("describe", **out)


def linear_regression(xs, ys):
    x, y = _vec(xs, "xs", 2), _vec(ys, "ys", 2)
    if len(x) != len(y):
        raise MathError("xs e ys devem ter o mesmo tamanho.")
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx == 0:
        raise MathError("Todos os x são iguais: regressão indefinida.")
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
    icpt = float(y.mean() - slope * x.mean())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float(((y - (slope * x + icpt)) ** 2).sum())
    return _out("linear_regression", slope=slope, intercept=icpt,
                r_squared=1.0 if ss_tot == 0 else 1 - ss_res / ss_tot, equation="y = slope·x + intercept")


METHODS = {f.__name__: f for f in (newton_raphson, bisection, trapezoid, simpson, numeric_derivative,
                                   lagrange, solve_linear, lu, qr, eig, describe, linear_regression)}


def run(method, params):
    if method not in METHODS:
        raise MathError(f"Método desconhecido: {method}. Disponíveis: {', '.join(sorted(METHODS))}.")
    try:
        return METHODS[method](**(params or {}))
    except TypeError as exc:
        raise MathError(f"Parâmetros inválidos para {method}: {exc}")
    except np.linalg.LinAlgError as exc:
        raise MathError(f"Erro de álgebra linear: {exc}")
