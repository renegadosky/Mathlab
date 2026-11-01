import math
from mathlab.engine import MathError
from mathlab import numeric as n


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except MathError:
        return True
    return False


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def test_newton():
    r = n.newton_raphson("x^2 - 2", 1)
    assert close(r["root"], math.sqrt(2)) and r["residual"] < 1e-10
    assert raises(n.newton_raphson, "x^2 + 1", 0)  # derivada nula em 0


def test_bisection():
    assert close(n.bisection("x^3 - x - 2", 1, 2)["root"], 1.5213797068045676, 1e-9)
    assert raises(n.bisection, "x^2 + 1", -1, 1)


def test_integracao():
    assert close(n.simpson("sin(x)", 0, math.pi, 100)["value"], 2.0, 1e-6)
    assert close(n.trapezoid("x^2", 0, 3, 3000)["value"], 9.0, 1e-5)
    assert raises(n.simpson, "x", 0, 1, 3)
    assert raises(n.trapezoid, "1/x", -1, 1, 2)  # x=0 indefinido


def test_derivada_numerica():
    assert close(n.numeric_derivative("sin(x)", 0)["value"], 1.0, 1e-8)


def test_lagrange_exato():
    r = n.lagrange([0, 1, 2], [1, 3, 7])
    assert r["polynomial"] == "x**2 + x + 1" and r["exact"]
    assert raises(n.lagrange, [1, 1], [2, 3])


def test_sistema_linear():
    r = n.solve_linear([[2, 1], [1, 3]], [3, 5])
    assert close(r["solution"][0], 0.8) and close(r["solution"][1], 1.4) and r["residual"] < 1e-12
    assert raises(n.solve_linear, [[1, 2], [2, 4]], [1, 2])


def test_lu_qr():
    A = [[2, 1, 1], [4, 3, 3], [8, 7, 9]]
    assert n.lu(A)["residual"] < 1e-12
    r = n.qr(A)
    assert r["residual"] < 1e-12 and r["orthogonality_error"] < 1e-12


def test_eig():
    r = n.eig([[2, 0], [0, 3]])
    assert sorted(r["eigenvalues"]) == [2.0, 3.0] and r["max_residual"] < 1e-12
    assert isinstance(n.eig([[0, -1], [1, 0]])["eigenvalues"][0], dict)  # complexos
    assert raises(n.eig, [[1, 2, 3]])


def test_estatistica():
    r = n.describe([2, 4, 4, 4, 5, 5, 7, 9])
    assert r["mean"] == 5 and close(r["variance_population"], 4)
    assert raises(n.describe, [])


def test_regressao():
    r = n.linear_regression([1, 2, 3, 4], [3, 5, 7, 9])
    assert close(r["slope"], 2) and close(r["intercept"], 1) and close(r["r_squared"], 1)
    assert raises(n.linear_regression, [1, 1], [2, 3])


def test_despacho():
    assert raises(n.run, "inexistente", {})
    assert raises(n.run, "simpson", {"foo": 1})
