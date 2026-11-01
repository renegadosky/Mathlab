import math
from mathlab.engine import MathError
from mathlab import physics as f, registry, plotting


def raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except MathError:
        return True
    return False


def v(r, k):
    return r["values"][k]["value"]


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1, abs(b))


def test_mecanica():
    r = f.kinematics_mruv(0, 2, 3)
    assert v(r, "v") == 6 and v(r, "s") == 9
    assert v(f.torricelli(0, 2, 9), "speed") == 6 and raises(f.torricelli, 0, -2, 9)
    assert close(v(f.circular_motion(2, speed=4), "centripetal_acceleration"), 8)
    assert raises(f.circular_motion, 2) and raises(f.newton_second, -1, 2)
    assert close(v(f.gravitation(1, 1, 1), "force"), 6.6743e-11)
    assert close(v(f.moment_of_inertia("hoop", 2, 3), "inertia"), 18) and raises(f.moment_of_inertia, "x", 1, 1)
    assert close(v(f.spring_oscillator(1, 4 * math.pi**2), "period"), 1)


def test_colisao():
    r = f.collision_1d(1, 2, 1, 0, 1)
    assert close(v(r, "v1_final"), 0) and close(v(r, "v2_final"), 2) and r["momentum_conserved"]
    assert close(v(r, "energy_lost"), 0)
    r = f.collision_1d(1, 2, 1, 0, 0)
    assert close(v(r, "v1_final"), 1) and close(v(r, "v2_final"), 1) and v(r, "energy_lost") > 0
    assert raises(f.collision_1d, 1, 1, 1, 1, 2)


def test_termo():
    assert close(v(f.ideal_gas(pressure=101325, moles=1, temperature=273.15), "volume"), 0.0224140, 1e-4)
    assert raises(f.ideal_gas, pressure=1, volume=1) and raises(f.ideal_gas, 1, 1, 1, 1)
    assert close(v(f.convert_temperature(100, "C", "F"), "temperature"), 212)
    assert raises(f.convert_temperature, -300, "C", "K")
    assert close(v(f.carnot_efficiency(300, 600), "efficiency"), 0.5) and raises(f.carnot_efficiency, 600, 300)
    assert close(v(f.heat(2, 4186, 10), "heat"), 83720)


def test_eletro():
    assert close(v(f.coulomb(1e-6, 1e-6, 1), "force"), 8.9875517923e-3)
    assert f.coulomb(1e-6, -1e-6, 1)["nature"] == "atrativa"
    assert v(f.ohm(voltage=12, resistance=4), "current") == 3 and raises(f.ohm, 1, 0, None)
    assert v(f.resistors([1, 2, 3]), "equivalent") == 6 and close(v(f.resistors([2, 2], "parallel"), "equivalent"), 1)
    assert close(v(f.capacitors([2, 2], "series"), "equivalent"), 1) and raises(f.resistors, [], "series")
    assert close(v(f.magnetic_field_wire(1, 1), "field"), 2e-7, 1e-6)
    assert close(v(f.faraday_emf(10, 0.2, 0.5), "emf"), -4)


def test_optica():
    assert close(v(f.wave(frequency=10, wavelength=2), "speed"), 20)
    assert close(v(f.snell(1, 1.5, 30), "theta2"), math.degrees(math.asin(0.5 / 1.5)))
    assert raises(f.snell, 1.5, 1, 60)
    r = f.thin_lens(10, 30)
    assert close(v(r, "image_distance"), 15) and close(v(r, "magnification"), -0.5) and r["image"] == "real"
    assert f.thin_lens(10, 5)["image"] == "virtual" and raises(f.thin_lens, 10, 10)


def test_registro_e_grafico():
    d = registry.describe()
    assert set(d) == {"physics", "programming", "numeric"} and len(d["physics"]) == len(f.METHODS)
    p = plotting.plot(["x^2 - 2", "tan(x)"], -3, 3, 600)
    zs = p["series"][0]["zeros_approx"]
    assert len(zs) == 2 and close(abs(zs[0]), math.sqrt(2), 1e-3)
    assert all(abs(math.tan(z)) < 1e-2 for z in p["series"][1]["zeros_approx"])  # sem falsos zeros em assíntotas
    assert raises(plotting.plot, ["x"], 1, 0) and raises(plotting.plot, ["y"], 0, 1) and raises(plotting.plot, [], 0, 1)
    assert plotting.plot(["sqrt(x)"], -1, 1)["series"][0]["y"][0] is None  # domínio inválido -> sem ponto
