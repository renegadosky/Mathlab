import math
from mathlab.engine import MathError
from mathlab import programming as p


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except MathError:
        return True
    return False


def close(a, b, tol=1e-9):
    return abs(a - b) < tol


def vclose(a, b, tol=1e-9):
    return all(close(x, y, tol) for x, y in zip(a, b))


def test_bases():
    assert p.convert_base("ff", 16, 2)["result"] == "11111111"
    assert p.convert_base("0xFF", 16, 10)["result"] == "255"
    assert p.convert_base("-101", 2, 10)["result"] == "-5"
    assert raises(p.convert_base, "12", 2, 10) and raises(p.convert_base, "1", 1, 10)


def test_bitwise():
    assert p.bitwise("and", 12, 10, 8)["unsigned"] == 8
    assert p.bitwise("not", 0, bits=8)["unsigned"] == 255 and p.bitwise("not", 0, bits=8)["signed"] == -1
    assert p.bitwise("shl", 1, 3, 8)["unsigned"] == 8
    assert raises(p.bitwise, "and", 300, 1, 8) and raises(p.bitwise, "foo", 1, 1)


def test_float():
    r = p.float_repr("0.1")
    assert r["hex"] == "0x3FB999999999999A" and r["representation_error"] != "0"
    assert p.float_repr("1", "single")["hex"] == "0x3F800000"
    assert raises(p.float_repr, "abc") and raises(p.float_repr, "1e300", "single")


def test_unidades():
    assert p.storage_convert(1, "GiB", "B")["exact_fraction"] == "1073741824"
    assert p.storage_convert(1, "GB", "MB")["value"] == 1000
    assert p.time_convert(90, "min", "h")["value"] == 1.5
    assert raises(p.time_convert, 1, "anos", "s")


def test_vetores():
    assert p.dot([1, 2, 3], [4, 5, 6])["value"] == 32
    assert p.cross([1, 0, 0], [0, 1, 0])["value"] == [0, 0, 1]
    assert close(p.angle([1, 0], [0, 1])["degrees"], 90)
    assert vclose(p.lerp([0, 0], [10, 20], 0.25)["value"], [2.5, 5])
    assert raises(p.cross, [1, 2], [3, 4]) and raises(p.angle, [0, 0], [1, 1]) and raises(p.dot, [1], [1, 2])


def test_polar():
    r = p.cartesian_to_polar(0, 2)
    assert close(r["r"], 2) and close(r["theta_deg"], 90)
    c = p.polar_to_cartesian(2, 90)
    assert close(c["x"], 0) and close(c["y"], 2)


def test_transformacoes():
    M = p.transform_matrix("rotation", 2, angle_deg=90)["matrix"]
    assert vclose(p.apply_matrix(M, [1, 0])["value"], [0, 1])
    T = p.transform_matrix("translation", 3, offset=[1, 2, 3])["matrix"]
    assert p.apply_matrix(T, [0, 0, 0])["value"] == [1, 2, 3]
    S = p.transform_matrix("scale", 2, factors=[2, 3])["matrix"]
    assert p.apply_matrix(S, [1, 1])["value"] == [2, 3]
    assert raises(p.apply_matrix, M, [1, 0, 0])


def test_quaternios():
    q = p.q_from_axis_angle([0, 0, 1], 90)["q"]
    assert vclose(p.q_rotate(q, [1, 0, 0])["value"], [0, 1, 0])
    q2 = p.q_multiply(q, q)["q"]  # 180° em z
    assert vclose(p.q_rotate(q2, [1, 0, 0])["value"], [-1, 0, 0])
    mid = p.slerp([1, 0, 0, 0], q, 0.5)["q"]  # 45°
    assert vclose(p.q_rotate(mid, [1, 0, 0])["value"], [math.sqrt(.5), math.sqrt(.5), 0])
    assert raises(p.q_rotate, [0, 0, 0, 0], [1, 0, 0]) and raises(p.q_from_axis_angle, [0, 0, 0], 10)


def test_colisoes():
    assert p.sphere_collision([0, 0], 1, [1.5, 0], 1)["colliding"]
    assert not p.sphere_collision([0, 0], 1, [3, 0], 1)["colliding"]
    assert p.sphere_collision([0, 0], 1, [2, 0], 1)["touching"]
    assert p.aabb_collision([0, 0], [2, 2], [1, 1], [3, 3])["colliding"]
    assert not p.aabb_collision([0, 0], [1, 1], [2, 2], [3, 3])["colliding"]
    assert raises(p.aabb_collision, [2, 2], [0, 0], [0, 0], [1, 1])


def test_big_o():
    assert p.big_o("3*n^2 + 5*n + 7")["class"] == "O(n^2)"
    assert p.big_o("n*log(n) + n")["class"] == "O(n log n)"
    assert p.big_o("2^n + n^3")["class"] == "O(2^n)"
    assert p.big_o("log(n) + 5")["class"] == "O(log n)"
    assert raises(p.big_o, "y + 1") and raises(p.big_o, "-n")
