import pytest
from fastapi.testclient import TestClient
from mathlab.engine import evaluate, MathError
from mathlab.api import app


def test_integral_verificada():
    r = evaluate("integrate(x^2, x)")
    assert r["result"] == "x**3/3" and r["verified"] is True and r["exact"]


def test_integral_definida():
    r = evaluate("integrate(x^2, (x, 0, 3))")
    assert r["result"] == "9" and r["verified"] is True


def test_derivada():
    assert evaluate("differentiate(sin(x)*x^2, x)")["result"] == "x**2*cos(x) + 2*x*sin(x)"


def test_limite():
    assert evaluate("limit(sin(x)/x, x, 0)")["result"] == "1"


def test_limite_lateral():
    assert evaluate("limit(1/x, x, 0, '+')")["result"] == "oo"


def test_equacao_segundo_grau_com_passos():
    r = evaluate("solve(x^2 - 5*x + 6 = 0, x)")
    assert r["result"] == "[2, 3]" and r["verified"] is True
    assert [s["title"] for s in r["steps"]][-1] == "Raízes"


def test_sem_solucao_real():
    r = evaluate("solve(x^2 + 1 = 0, x)")
    assert "complexa" in r["note"]


def test_matriz():
    assert evaluate("det(matrix([[1,2],[3,4]]))")["result"] == "-2"
    assert evaluate("eigenvals(matrix([[2,0],[0,3]]))")["exact"]


def test_matriz_singular():
    with pytest.raises(MathError):
        evaluate("inv(matrix([[1,2],[2,4]]))")


def test_aproximado_vs_exato():
    assert evaluate("sqrt(2)")["exact"] is True
    assert evaluate("N(sqrt(2), 20)")["exact"] is False


@pytest.mark.parametrize("bad", ["1/0", "", "2 +", "__import__('os').system('ls')",
                                 "open('x')", "x.__class__", "inv(matrix([[0]]))"])
def test_entradas_invalidas(bad):
    with pytest.raises(MathError):
        evaluate(bad)


def test_api():
    c = TestClient(app)
    assert c.post("/api/evaluate", json={"expression": "2+2"}).json()["result"] == "4"
    assert c.post("/api/evaluate", json={"expression": "1/0"}).status_code == 422
