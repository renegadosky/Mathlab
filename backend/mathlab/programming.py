"""Matemática para desenvolvedores. Conversões exatas (int/Fraction); geometria em float64."""
import math
import struct
from decimal import Decimal, localcontext
from fractions import Fraction
import numpy as np
import sympy as sp
from .engine import MathError, parse_function

DIG = "0123456789abcdefghijklmnopqrstuvwxyz"


# ---------- bases e bitwise ----------
def _base(b):
    if isinstance(b, bool) or not isinstance(b, int) or not 2 <= b <= 36:
        raise MathError("A base deve ser um inteiro entre 2 e 36.")
    return b


def _to_str(n, b):
    if n == 0:
        return "0"
    out, m = [], abs(n)
    while m:
        m, r = divmod(m, b)
        out.append(DIG[r])
    return ("-" if n < 0 else "") + "".join(reversed(out))


def convert_base(value, from_base, to_base):
    fb, tb = _base(from_base), _base(to_base)
    s = str(value).strip().lower().replace("_", "")
    neg = False
    if s[:1] in ("+", "-"):
        neg, s = s[0] == "-", s[1:]
    for pre, base in (("0x", 16), ("0b", 2), ("0o", 8)):
        if fb == base and s.startswith(pre):
            s = s[2:]
    if not s or len(s) > 4000 or any(c not in DIG[:fb] for c in s):
        raise MathError(f"'{value}' não é um número válido na base {fb}.")
    n = -int(s, fb) if neg else int(s, fb)
    return {"method": "convert_base", "exact": True, "decimal": str(n), "result": _to_str(n, tb).upper(),
            "binary": _to_str(n, 2), "octal": _to_str(n, 8), "hexadecimal": _to_str(n, 16).upper()}


def bitwise(op, a, b=0, bits=32):
    if bits not in (8, 16, 32, 64):
        raise MathError("bits deve ser 8, 16, 32 ou 64.")
    for v in (a, b):
        if isinstance(v, bool) or not isinstance(v, int) or not -(1 << (bits - 1)) <= v < (1 << bits):
            raise MathError(f"Operando fora do intervalo de {bits} bits.")
    mask = (1 << bits) - 1
    ua, ub = a & mask, b & mask
    if op in ("shl", "shr") and not 0 <= b <= bits:
        raise MathError("Deslocamento deve estar entre 0 e a largura em bits.")
    ops = {"and": lambda: ua & ub, "or": lambda: ua | ub, "xor": lambda: ua ^ ub,
           "not": lambda: ~ua & mask, "shl": lambda: (ua << b) & mask, "shr": lambda: ua >> b}
    if op not in ops:
        raise MathError(f"Operação desconhecida: {op}. Use {', '.join(ops)}.")
    r = ops[op]()
    return {"method": "bitwise", "exact": True, "bits": bits, "unsigned": r,
            "signed": r - (1 << bits) if r >> (bits - 1) else r,
            "hex": f"0x{r:0{bits // 4}X}", "binary": f"{r:0{bits}b}"}


def float_repr(value, precision="double"):
    fmt = {"double": (">d", ">Q", 11, 52, 1023), "single": (">f", ">I", 8, 23, 127)}
    if precision not in fmt:
        raise MathError("precision deve ser 'single' ou 'double'.")
    f, u, eb, mb, bias = fmt[precision]
    try:
        d = Decimal(str(value).strip())
        x = float(d) if d.is_finite() else float(str(value))
        raw = struct.pack(f, x)
    except OverflowError:
        raise MathError("Valor fora da faixa desta precisão.")
    except Exception:
        raise MathError("Valor numérico inválido.")
    bitsv = int.from_bytes(raw, "big")
    total = eb + mb + 1
    s = f"{bitsv:0{total}b}"
    e, m = int(s[1:1 + eb], 2), int(s[1 + eb:], 2)
    kind = ("nan" if m else "infinity") if e == (1 << eb) - 1 else ("zero" if not e and not m else
            "subnormal" if not e else "normal")
    stored = struct.unpack(f, raw)[0]
    out = {"method": "float_repr", "exact": True, "precision": precision, "class": kind, "sign": s[0],
           "exponent_bits": s[1:1 + eb], "mantissa_bits": s[1 + eb:], "hex": f"0x{bitsv:0{total // 4}X}",
           "biased_exponent": e, "unbiased_exponent": e - bias if kind in ("normal", "subnormal") else None}
    if math.isfinite(stored):
        with localcontext() as c:
            c.prec = 80
            out["stored_value"] = str(Decimal(stored))
            out["representation_error"] = str(Decimal(stored) - d) if d.is_finite() else None
    return out


# ---------- unidades ----------
_STORAGE = {"bit": Fraction(1, 8), "B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12, "PB": 10**15,
            "KiB": 2**10, "MiB": 2**20, "GiB": 2**30, "TiB": 2**40, "PiB": 2**50}
_TIME = {"ns": Fraction(1, 10**9), "us": Fraction(1, 10**6), "ms": Fraction(1, 1000), "s": 1, "min": 60,
         "h": 3600, "d": 86400, "week": 604800}


def _convert(value, a, b, table, name):
    if a not in table or b not in table:
        raise MathError(f"Unidade de {name} desconhecida. Disponíveis: {', '.join(table)}.")
    try:
        v = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise MathError("Valor numérico inválido.")
    r = v * Fraction(table[a]) / Fraction(table[b])
    return {"exact": True, "value": float(r), "exact_fraction": str(r), "from": a, "to": b}


def storage_convert(value, from_unit, to_unit):
    return {"method": "storage_convert", **_convert(value, from_unit, to_unit, _STORAGE, "armazenamento"),
            "note": "KB/MB/GB = potências de 10; KiB/MiB/GiB = potências de 2."}


def time_convert(value, from_unit, to_unit):
    return {"method": "time_convert", **_convert(value, from_unit, to_unit, _TIME, "tempo")}


# ---------- vetores ----------
def _v(a, name="vetor", dim=None):
    try:
        v = np.array(a, dtype=float)
    except (ValueError, TypeError):
        raise MathError(f"{name} deve ser uma lista numérica.")
    if v.ndim != 1 or len(v) == 0 or not np.all(np.isfinite(v)) or (dim and len(v) != dim):
        raise MathError(f"{name} inválido" + (f" (esperado {dim} componentes)." if dim else "."))
    return v


def _pair(a, b):
    a, b = _v(a, "a"), _v(b, "b")
    if len(a) != len(b):
        raise MathError("Os vetores devem ter a mesma dimensão.")
    return a, b


def dot(a, b):
    a, b = _pair(a, b)
    return {"method": "dot", "value": float(a @ b), "formula": r"a\cdot b=\sum a_i b_i"}


def cross(a, b):
    a, b = _v(a, "a", 3), _v(b, "b", 3)
    return {"method": "cross", "value": np.cross(a, b).tolist(), "formula": r"a\times b"}


def norm(a):
    return {"method": "norm", "value": float(np.linalg.norm(_v(a)))}


def angle(a, b):
    a, b = _pair(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise MathError("Ângulo indefinido para vetor nulo.")
    r = math.acos(max(-1.0, min(1.0, float(a @ b) / (na * nb))))
    return {"method": "angle", "radians": r, "degrees": math.degrees(r)}


def distance(a, b):
    a, b = _pair(a, b)
    return {"method": "distance", "value": float(np.linalg.norm(a - b)), "formula": r"\|a-b\|_2"}


def lerp(a, b, t):
    a, b = _pair(a, b)
    return {"method": "lerp", "value": (a + (b - a) * float(t)).tolist(), "formula": r"a+t(b-a)",
            "note": "Extrapolação (t fora de [0,1])." if not 0 <= float(t) <= 1 else None}


def cartesian_to_polar(x, y):
    t = math.atan2(float(y), float(x))
    return {"method": "cartesian_to_polar", "r": math.hypot(float(x), float(y)), "theta_rad": t,
            "theta_deg": math.degrees(t)}


def polar_to_cartesian(r, theta_deg):
    if float(r) < 0:
        raise MathError("r deve ser ≥ 0.")
    t = math.radians(float(theta_deg))
    return {"method": "polar_to_cartesian", "x": float(r) * math.cos(t), "y": float(r) * math.sin(t)}


# ---------- transformações ----------
def transform_matrix(kind, dim=3, angle_deg=None, axis="z", factors=None, offset=None):
    if dim not in (2, 3):
        raise MathError("dim deve ser 2 ou 3.")
    M = np.eye(dim + 1)
    if kind == "rotation":
        if angle_deg is None:
            raise MathError("Informe angle_deg.")
        c, s = math.cos(math.radians(float(angle_deg))), math.sin(math.radians(float(angle_deg)))
        if dim == 2 or axis == "z":
            M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, -s, s, c
        elif axis == "x":
            M[1, 1], M[1, 2], M[2, 1], M[2, 2] = c, -s, s, c
        elif axis == "y":
            M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c
        else:
            raise MathError("axis deve ser x, y ou z.")
    elif kind == "scale":
        M[np.arange(dim), np.arange(dim)] = _v(factors, "factors", dim)
    elif kind == "translation":
        M[:dim, dim] = _v(offset, "offset", dim)
    else:
        raise MathError("kind deve ser rotation, scale ou translation.")
    M[np.abs(M) < 1e-15] = 0.0
    return {"method": "transform_matrix", "matrix": M.tolist(), "homogeneous": True}


def apply_matrix(matrix, point):
    p = _v(point, "point")
    try:
        M = np.array(matrix, dtype=float)
    except (ValueError, TypeError):
        raise MathError("matrix inválida.")
    if M.shape != (len(p) + 1, len(p) + 1):
        raise MathError(f"Para um ponto de dimensão {len(p)}, a matriz deve ser {len(p)+1}×{len(p)+1}.")
    return {"method": "apply_matrix", "value": (M @ np.append(p, 1.0))[:len(p)].tolist()}


# ---------- quatérnios (w, x, y, z) ----------
def _q(q, name="q"):
    q = _v(q, name, 4)
    n = np.linalg.norm(q)
    if n == 0:
        raise MathError("Quatérnio nulo.")
    return q / n


def q_from_axis_angle(axis, angle_deg):
    ax = _v(axis, "axis", 3)
    if np.linalg.norm(ax) == 0:
        raise MathError("Eixo nulo.")
    ax = ax / np.linalg.norm(ax)
    h = math.radians(float(angle_deg)) / 2
    return {"method": "q_from_axis_angle", "q": [math.cos(h), *(math.sin(h) * ax).tolist()], "order": "w,x,y,z"}


def _mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2])


def q_multiply(a, b):
    return {"method": "q_multiply", "q": _mul(_v(a, "a", 4), _v(b, "b", 4)).tolist(), "note": "Produto de Hamilton."}


def q_rotate(q, v):
    q, v = _q(q), _v(v, "v", 3)
    conj = q * np.array([1, -1, -1, -1])
    return {"method": "q_rotate", "value": _mul(_mul(q, np.append(0.0, v)), conj)[1:].tolist(), "formula": r"v'=qvq^{*}"}


def slerp(q0, q1, t):
    a, b = _q(q0, "q0"), _q(q1, "q1")
    t = float(t)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d  # caminho mais curto
    if d > 0.9995:
        r = a + t * (b - a)
        r /= np.linalg.norm(r)
        note = "Quatérnios quase idênticos: usado lerp normalizado."
    else:
        th = math.acos(d)
        r = (math.sin((1 - t) * th) * a + math.sin(t * th) * b) / math.sin(th)
        note = None
    return {"method": "slerp", "q": r.tolist(), "note": note, "formula": r"\frac{\sin((1-t)\theta)q_0+\sin(t\theta)q_1}{\sin\theta}"}


# ---------- colisões ----------
def sphere_collision(c1, r1, c2, r2):
    a, b = _pair(c1, c2)
    if float(r1) <= 0 or float(r2) <= 0:
        raise MathError("Os raios devem ser positivos.")
    d, s = float(np.linalg.norm(a - b)), float(r1) + float(r2)
    return {"method": "sphere_collision", "colliding": d < s - 1e-12, "touching": abs(d - s) <= 1e-12,
            "distance": d, "penetration": max(0.0, s - d)}


def aabb_collision(min1, max1, min2, max2):
    a0, a1, b0, b1 = _v(min1, "min1"), _v(max1, "max1"), _v(min2, "min2"), _v(max2, "max2")
    if not (len(a0) == len(a1) == len(b0) == len(b1)) or np.any(a0 > a1) or np.any(b0 > b1):
        raise MathError("Caixas inválidas: dimensões iguais e min ≤ max.")
    return {"method": "aabb_collision", "colliding": bool(np.all(a0 <= b1) and np.all(b0 <= a1))}


# ---------- Big O ----------
def big_o(expression):
    n = sp.Symbol("n", positive=True)
    e = parse_function(expression, "n").subs(sp.Symbol("n"), n)
    note = "Análise assintótica (n→∞) de uma função de custo informada; não analisa código-fonte."
    if e.is_polynomial(n):
        p = sp.Poly(e, n)
        if p.LC() <= 0:
            raise MathError("O termo dominante deve ser positivo para uma função de custo.")
        d = p.degree()
        return {"method": "big_o", "exact": True, "class": "O(1)" if d == 0 else f"O(n^{d})" if d > 1 else "O(n)",
                "theta": True, "note": note}
    for g, name in ((1, "O(1)"), (sp.log(n), "O(log n)"), (n, "O(n)"), (n * sp.log(n), "O(n log n)"),
                    (n**2, "O(n^2)"), (n**3, "O(n^3)"), (2**n, "O(2^n)"), (sp.factorial(n), "O(n!)")):
        L = sp.limit(e / g, n, sp.oo)
        if L.is_finite and L != 0 and not L.has(sp.nan):
            return {"method": "big_o", "exact": True, "class": name, "theta": True, "limit_ratio": str(L), "note": note}
    raise MathError("Classe de crescimento não reconhecida para esta expressão.")


METHODS = {f.__name__: f for f in (
    convert_base, bitwise, float_repr, storage_convert, time_convert, dot, cross, norm, angle, distance, lerp,
    cartesian_to_polar, polar_to_cartesian, transform_matrix, apply_matrix, q_from_axis_angle, q_multiply,
    q_rotate, slerp, sphere_collision, aabb_collision, big_o)}


def run(method, params):
    if method not in METHODS:
        raise MathError(f"Método desconhecido: {method}. Disponíveis: {', '.join(sorted(METHODS))}.")
    try:
        return METHODS[method](**(params or {}))
    except TypeError as exc:
        raise MathError(f"Parâmetros inválidos para {method}: {exc}")
