import numpy as np
import sympy as sp
from .engine import MathError, parse_function

X = sp.Symbol("x")


def plot(expressions, xmin, xmax, points=600):
    if not isinstance(expressions, list) or not 1 <= len(expressions) <= 8:
        raise MathError("Informe de 1 a 8 funções.")
    try:
        a, b, n = float(xmin), float(xmax), int(points)
    except (TypeError, ValueError):
        raise MathError("Domínio inválido.")
    if not (np.isfinite(a) and np.isfinite(b) and a < b):
        raise MathError("É necessário xmin < xmax (finitos).")
    if not 10 <= n <= 5000:
        raise MathError("points deve estar entre 10 e 5000.")
    xs = np.linspace(a, b, n)
    series = []
    for text in expressions:
        e = parse_function(text)
        f = sp.lambdify(X, e, "numpy")
        with np.errstate(all="ignore"):
            y = np.asarray(f(xs), dtype=complex) * np.ones(n)
        y = np.where(np.abs(y.imag) < 1e-12, y.real, np.nan)
        y = np.where(np.isfinite(y), y, np.nan)
        zeros = []
        for i in range(n - 1):
            y0, y1 = y[i], y[i + 1]
            if np.isfinite(y0) and np.isfinite(y1) and y0 * y1 < 0:
                z = xs[i] - y0 * (xs[i + 1] - xs[i]) / (y1 - y0)
                with np.errstate(all="ignore"):
                    fz = np.real(complex(f(z)))
                if np.isfinite(fz) and abs(fz) < min(abs(y0), abs(y1)):  # descarta assíntotas
                    zeros.append(float(z))
        series.append({"expression": text, "y": [None if np.isnan(v) else float(v) for v in y],
                       "zeros_approx": zeros})
    return {"x": xs.tolist(), "series": series}
