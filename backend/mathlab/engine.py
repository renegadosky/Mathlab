"""Núcleo matemático do MathLab: parser seguro (AST) + despacho para SymPy.

Sem regex frágil: a entrada é validada pela AST do Python (lista branca de nós
e de funções) e só então avaliada pelo parser do SymPy.
"""
import ast
from dataclasses import dataclass, field, asdict
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations


class MathError(Exception):
    pass


@dataclass
class Outcome:
    value: object
    steps: list = field(default_factory=list)
    verified: bool | None = None
    note: str | None = None


def _quadratic_steps(expr, x):
    try:
        p = sp.Poly(sp.expand(expr), x)
    except Exception:
        return []
    if p.degree() != 2 or not all(c.is_number for c in p.all_coeffs()):
        return []
    a, b, c = p.all_coeffs()
    d = sp.simplify(b**2 - 4*a*c)
    roots = sp.solve(expr, x)
    return [
        {"title": "Forma geral", "latex": f"{sp.latex(a)}x^2 + {sp.latex(b)}x + {sp.latex(c)} = 0"},
        {"title": "Fórmula de Bhaskara", "latex": r"x = \frac{-b \pm \sqrt{\Delta}}{2a},\ \Delta = b^2 - 4ac"},
        {"title": "Discriminante", "latex": rf"\Delta = ({sp.latex(b)})^2 - 4({sp.latex(a)})({sp.latex(c)}) = {sp.latex(d)}"},
        {"title": "Raízes", "latex": ", ".join(sp.latex(r) for r in roots)},
    ]


def _solve(eq, *vars_):
    sols = sp.solve(eq, *vars_)
    out = Outcome(sols)
    if isinstance(eq, sp.core.relational.Relational) and not isinstance(eq, sp.Eq):
        return out
    if len(vars_) == 1 and isinstance(sols, list):
        x = vars_[0]
        lhs = eq.lhs - eq.rhs if isinstance(eq, sp.Eq) else eq
        out.steps = _quadratic_steps(lhs, x)
        if sols:
            out.verified = all(sp.simplify(lhs.subs(x, s)) == 0 for s in sols)
            if not any(s.is_real for s in sols):
                out.note = "Sem solução real; existem apenas soluções complexas."
        else:
            out.note = "Sem solução."
    return out


def _integrate(f, *args):
    f = sp.sympify(f)
    res = sp.integrate(f, *args)
    if res.has(sp.Integral):
        raise MathError("Não foi encontrada forma fechada para a integral.")
    out = Outcome(res)
    if len(args) == 1 and not isinstance(args[0], tuple):
        out.verified = sp.simplify(sp.diff(res, args[0]) - f) == 0
        out.note = "Constante de integração omitida (+ C)."
    else:
        try:
            ref = sp.Integral(f, *args).evalf()
            out.verified = abs(complex(res.evalf() - ref)) < 1e-8
        except Exception:
            pass
    return out


def _limit(f, x, x0, d="+-"):
    res = sp.limit(f, x, x0, d)
    if res.has(sp.Limit):
        raise MathError("Não foi possível determinar o limite.")
    return res


def _inv(m):
    if m.det() == 0:
        raise MathError("Matriz singular: não possui inversa.")
    return m.inv()


LOCAL = {
    "integrate": _integrate, "differentiate": lambda f, *v: sp.diff(f, *v),
    "diff": lambda f, *v: sp.diff(f, *v), "limit": _limit, "solve": _solve,
    "matrix": sp.Matrix, "det": lambda m: m.det(), "inv": _inv,
    "transpose": lambda m: m.T, "eigenvals": lambda m: m.eigenvals(),
    "simplify": sp.simplify, "factor": sp.factor, "expand": sp.expand,
    "N": lambda x, n=15: sp.N(x, n),
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "asin": sp.asin, "acos": sp.acos,
    "atan": sp.atan, "exp": sp.exp, "ln": sp.log, "log": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "factorial": sp.factorial, "binomial": sp.binomial,
    "pi": sp.pi, "e": sp.E, "i": sp.I, "oo": sp.oo,
    "Eq": sp.Eq, "Lt": sp.Lt, "Le": sp.Le, "Gt": sp.Gt, "Ge": sp.Ge,
}
GLOBAL = {"Symbol": sp.Symbol, "Integer": sp.Integer, "Float": sp.Float, "Rational": sp.Rational}
_CMP = {ast.Eq: "Eq", ast.Lt: "Lt", ast.LtE: "Le", ast.Gt: "Gt", ast.GtE: "Ge"}
_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)


class _Compare(ast.NodeTransformer):
    def visit_Compare(self, node):
        op = _CMP.get(type(node.ops[0]))
        if len(node.ops) != 1 or op is None:
            raise MathError("Comparação não suportada.")
        self.generic_visit(node)
        return ast.Call(func=ast.Name(id=op, ctx=ast.Load()),
                        args=[node.left, node.comparators[0]], keywords=[])


def _prepare(text):
    text = text.replace("^", "**")
    out = []
    for i, ch in enumerate(text):  # '=' isolado -> '=='
        prev = text[i-1] if i else ""
        nxt = text[i+1] if i + 1 < len(text) else ""
        out.append("==" if ch == "=" and prev not in "<>!=" and nxt != "=" else ch)
    return "".join(out).replace("====", "==")


def _validate(text):
    try:
        tree = ast.parse(_prepare(text), mode="eval")
    except SyntaxError:
        raise MathError("Expressão inválida (erro de sintaxe). Use * para multiplicar.")
    tree = ast.fix_missing_locations(_Compare().visit(tree))
    for n in ast.walk(tree):
        if isinstance(n, (ast.Expression, ast.List, ast.Tuple, ast.Load, ast.UnaryOp,
                          ast.UAdd, ast.USub, *_BIN)):
            continue
        if isinstance(n, ast.BinOp):
            continue
        if isinstance(n, ast.Constant) and (isinstance(n.value, (int, float)) or n.value in ("+", "-")):
            continue
        if isinstance(n, ast.Name) and not n.id.startswith("_"):
            continue
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in LOCAL and not n.keywords:
            continue
        raise MathError(f"Construção não permitida na expressão: {type(n).__name__}.")
    return tree


def _has_float(v):
    if isinstance(v, (list, tuple, set)):
        return any(_has_float(i) for i in v)
    if isinstance(v, dict):
        return any(_has_float(i) for i in v.values())
    return bool(getattr(v, "has", lambda *_: False)(sp.Float))


def _has_bad(v):
    if isinstance(v, (list, tuple, set)):
        return any(_has_bad(i) for i in v)
    if isinstance(v, dict):
        return any(_has_bad(i) for i in v.values())
    return bool(getattr(v, "has", lambda *_: False)(sp.zoo, sp.nan))


def evaluate(text: str) -> dict:
    if not text or not text.strip():
        raise MathError("Expressão vazia.")
    tree = _validate(text.strip())
    try:
        res = parse_expr(ast.unparse(tree), local_dict=LOCAL, global_dict=GLOBAL,
                         transformations=standard_transformations)
    except ZeroDivisionError:
        raise MathError("Divisão por zero.")
    except MathError:
        raise
    except Exception as exc:
        raise MathError(f"Não foi possível calcular: {exc}")
    out = res if isinstance(res, Outcome) else Outcome(res)
    if _has_bad(out.value):
        raise MathError("Resultado indefinido (divisão por zero ou forma indeterminada).")
    try:
        latex = sp.latex(out.value)
    except Exception:
        latex = str(out.value)
    return {"input": text, "result": str(out.value), "latex": latex,
            "exact": not _has_float(out.value), "steps": out.steps,
            "verified": out.verified, "note": out.note}


def parse_function(text: str, var: str = "x"):
    """Converte texto em expressão SymPy de uma variável, usando o parser seguro."""
    if not text or not text.strip():
        raise MathError("Função vazia.")
    tree = _validate(text.strip())
    try:
        expr = parse_expr(ast.unparse(tree), local_dict=LOCAL, global_dict=GLOBAL,
                          transformations=standard_transformations)
    except Exception as exc:
        raise MathError(f"Função inválida: {exc}")
    if not isinstance(expr, sp.Expr):
        raise MathError("A entrada precisa ser uma expressão em função de " + var + ".")
    extra = {str(s) for s in expr.free_symbols} - {var}
    if extra:
        raise MathError(f"Símbolos desconhecidos: {', '.join(sorted(extra))}. Use apenas '{var}'.")
    return expr
