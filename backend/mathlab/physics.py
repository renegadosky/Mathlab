"""Física: entradas e saídas em unidades SI. Cada ferramenta devolve valores com unidade e a fórmula (LaTeX)."""
import math
from .engine import MathError

G, G0, K, MU0, R_GAS = 6.67430e-11, 9.80665, 8.9875517923e9, 1.25663706212e-6, 8.314462618


def _n(x, name):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise MathError(f"{name} deve ser numérico.")
    if not math.isfinite(v):
        raise MathError(f"{name} deve ser finito.")
    return v


def _p(x, name):
    v = _n(x, name)
    if v <= 0:
        raise MathError(f"{name} deve ser positivo.")
    return v


def _m(formula, **vals):
    return {"exact": False, "formula": formula,
            "values": {k: {"value": v[0], "unit": v[1]} for k, v in vals.items()}}


def _one_missing(**kw):
    missing = [k for k, v in kw.items() if v is None]
    if len(missing) != 1:
        raise MathError(f"Deixe exatamente um destes vazio (será calculado): {', '.join(kw)}.")
    return missing[0]


# ---- mecânica ----
def kinematics_mruv(v0, a, t):
    """MRUV: v = v0 + a t; s = v0 t + a t²/2."""
    v0, a, t = _n(v0, "v0"), _n(a, "a"), _n(t, "t")
    if t < 0:
        raise MathError("t não pode ser negativo.")
    return _m(r"v=v_0+at,\ s=v_0t+\frac{at^2}{2}", v=(v0 + a * t, "m/s"), s=(v0 * t + a * t * t / 2, "m"))


def torricelli(v0, a, s):
    """Equação de Torricelli: v² = v0² + 2 a s."""
    v0, a, s = _n(v0, "v0"), _n(a, "a"), _n(s, "s")
    v2 = v0 * v0 + 2 * a * s
    if v2 < 0:
        raise MathError("Sem solução real: o corpo não alcança esse deslocamento.")
    return _m(r"v^2=v_0^2+2as", speed=(math.sqrt(v2), "m/s"))


def circular_motion(radius, speed=None, period=None):
    """Movimento circular uniforme; informe speed OU period."""
    r = _p(radius, "radius")
    if (speed is None) == (period is None):
        raise MathError("Informe speed ou period (apenas um).")
    v = _p(speed, "speed") if speed is not None else 2 * math.pi * r / _p(period, "period")
    T = 2 * math.pi * r / v
    return _m(r"a_c=\frac{v^2}{r},\ T=\frac{2\pi r}{v}", centripetal_acceleration=(v * v / r, "m/s²"),
              period=(T, "s"), frequency=(1 / T, "Hz"), angular_velocity=(v / r, "rad/s"), speed=(v, "m/s"))


def newton_second(mass, acceleration):
    """Segunda lei de Newton: F = m a."""
    return _m(r"F=ma", force=(_p(mass, "mass") * _n(acceleration, "acceleration"), "N"))


def weight(mass, g=G0):
    """Peso: P = m g."""
    return _m(r"P=mg", weight=(_p(mass, "mass") * _p(g, "g"), "N"))


def work(force, distance, angle_deg=0):
    """Trabalho: W = F d cosθ."""
    return _m(r"W=Fd\cos\theta", work=(_n(force, "force") * _n(distance, "distance") *
                                      math.cos(math.radians(_n(angle_deg, "angle_deg"))), "J"))


def kinetic_energy(mass, speed):
    """Energia cinética: Ec = m v²/2."""
    return _m(r"E_c=\frac{mv^2}{2}", kinetic_energy=(_p(mass, "mass") * _n(speed, "speed") ** 2 / 2, "J"))


def potential_energy(mass, height, g=G0):
    """Energia potencial gravitacional: Ep = m g h."""
    return _m(r"E_p=mgh", potential_energy=(_p(mass, "mass") * _p(g, "g") * _n(height, "height"), "J"))


def collision_1d(m1, v1, m2, v2, restitution=1.0):
    """Colisão unidimensional com coeficiente de restituição e (0 = perfeitamente inelástica, 1 = elástica)."""
    m1, m2, v1, v2, e = _p(m1, "m1"), _p(m2, "m2"), _n(v1, "v1"), _n(v2, "v2"), _n(restitution, "restitution")
    if not 0 <= e <= 1:
        raise MathError("restitution deve estar em [0, 1].")
    p = m1 * v1 + m2 * v2
    u1, u2 = (p + m2 * e * (v2 - v1)) / (m1 + m2), (p + m1 * e * (v1 - v2)) / (m1 + m2)
    ke0, ke1 = (m1 * v1**2 + m2 * v2**2) / 2, (m1 * u1**2 + m2 * u2**2) / 2
    out = _m(r"m_1v_1+m_2v_2=m_1v_1'+m_2v_2'", v1_final=(u1, "m/s"), v2_final=(u2, "m/s"),
             momentum=(p, "kg·m/s"), kinetic_energy_before=(ke0, "J"), kinetic_energy_after=(ke1, "J"),
             energy_lost=(ke0 - ke1, "J"))
    out["momentum_conserved"] = abs(m1 * u1 + m2 * u2 - p) <= 1e-9 * (1 + abs(p))
    return out


def gravitation(m1, m2, r):
    """Lei da gravitação universal: F = G m1 m2 / r²."""
    return _m(r"F=\frac{Gm_1m_2}{r^2}", force=(G * _p(m1, "m1") * _p(m2, "m2") / _p(r, "r") ** 2, "N"))


def torque(force, lever_arm, angle_deg=90):
    """Torque: τ = F r sinθ."""
    return _m(r"\tau=Fr\sin\theta", torque=(_n(force, "force") * _p(lever_arm, "lever_arm") *
                                           math.sin(math.radians(_n(angle_deg, "angle_deg"))), "N·m"))


_INERTIA = {"hoop": 1, "solid_cylinder": 1 / 2, "solid_sphere": 2 / 5, "hollow_sphere": 2 / 3,
            "rod_center": 1 / 12, "rod_end": 1 / 3}


def moment_of_inertia(shape, mass, size):
    """Momento de inércia I = k m s² (size = raio ou comprimento). Formas: hoop, solid_cylinder, solid_sphere, hollow_sphere, rod_center, rod_end."""
    if shape not in _INERTIA:
        raise MathError(f"Forma desconhecida. Use: {', '.join(_INERTIA)}.")
    return _m(r"I=k\,m\,s^2", inertia=(_INERTIA[shape] * _p(mass, "mass") * _p(size, "size") ** 2, "kg·m²"),
              k=(_INERTIA[shape], "adimensional"))


def spring_oscillator(mass, spring_constant):
    """MHS massa-mola: T = 2π√(m/k)."""
    T = 2 * math.pi * math.sqrt(_p(mass, "mass") / _p(spring_constant, "spring_constant"))
    return _m(r"T=2\pi\sqrt{m/k}", period=(T, "s"), frequency=(1 / T, "Hz"), angular_frequency=(2 * math.pi / T, "rad/s"))


def pendulum_period(length, g=G0):
    """Pêndulo simples (pequenas oscilações): T = 2π√(L/g)."""
    out = _m(r"T=2\pi\sqrt{L/g}", period=(2 * math.pi * math.sqrt(_p(length, "length") / _p(g, "g")), "s"))
    out["note"] = "Aproximação válida para pequenos ângulos."
    return out


# ---- termodinâmica ----
def ideal_gas(pressure=None, volume=None, moles=None, temperature=None):
    """Gás ideal PV = nRT (SI: Pa, m³, mol, K). Deixe um parâmetro vazio."""
    d = {"pressure": pressure, "volume": volume, "moles": moles, "temperature": temperature}
    miss = _one_missing(**d)
    v = {k: _p(x, k) for k, x in d.items() if x is not None}
    calc = {"pressure": lambda: v["moles"] * R_GAS * v["temperature"] / v["volume"],
            "volume": lambda: v["moles"] * R_GAS * v["temperature"] / v["pressure"],
            "moles": lambda: v["pressure"] * v["volume"] / (R_GAS * v["temperature"]),
            "temperature": lambda: v["pressure"] * v["volume"] / (v["moles"] * R_GAS)}
    unit = {"pressure": "Pa", "volume": "m³", "moles": "mol", "temperature": "K"}
    return _m(r"PV=nRT", **{miss: (calc[miss](), unit[miss])})


def thermal_expansion(alpha, length0, delta_t):
    """Dilatação linear: ΔL = α L0 ΔT (alpha em 1/K)."""
    return _m(r"\Delta L=\alpha L_0\Delta T", delta_length=(_n(alpha, "alpha") * _p(length0, "length0") * _n(delta_t, "delta_t"), "m"))


def heat(mass, specific_heat, delta_t):
    """Calorimetria: Q = m c ΔT (c em J/(kg·K))."""
    return _m(r"Q=mc\Delta T", heat=(_p(mass, "mass") * _p(specific_heat, "specific_heat") * _n(delta_t, "delta_t"), "J"))


def carnot_efficiency(t_cold, t_hot):
    """Rendimento de Carnot: η = 1 − Tc/Th (kelvin)."""
    tc, th = _p(t_cold, "t_cold"), _p(t_hot, "t_hot")
    if tc >= th:
        raise MathError("É necessário t_cold < t_hot.")
    return _m(r"\eta=1-\frac{T_c}{T_h}", efficiency=(1 - tc / th, "adimensional"))


def entropy_change(heat_joules, temperature):
    """Variação de entropia (processo reversível isotérmico): ΔS = Q/T."""
    return _m(r"\Delta S=\frac{Q}{T}", delta_entropy=(_n(heat_joules, "heat_joules") / _p(temperature, "temperature"), "J/K"))


def convert_temperature(value, from_unit, to_unit):
    """Conversão entre C, F e K (rejeita valores abaixo do zero absoluto)."""
    v = _n(value, "value")
    to_k = {"C": lambda x: x + 273.15, "F": lambda x: (x - 32) * 5 / 9 + 273.15, "K": lambda x: x}
    from_k = {"C": lambda k: k - 273.15, "F": lambda k: (k - 273.15) * 9 / 5 + 32, "K": lambda k: k}
    if from_unit not in to_k or to_unit not in from_k:
        raise MathError("Unidades válidas: C, F, K.")
    k = to_k[from_unit](v)
    if k < 0:
        raise MathError("Temperatura abaixo do zero absoluto.")
    return _m(r"T_K=T_C+273{,}15", temperature=(from_k[to_unit](k), to_unit))


# ---- eletricidade e magnetismo ----
def coulomb(q1, q2, r):
    """Lei de Coulomb: F = k q1 q2 / r² (positivo = repulsão)."""
    F = K * _n(q1, "q1") * _n(q2, "q2") / _p(r, "r") ** 2
    out = _m(r"F=\frac{k q_1q_2}{r^2}", force=(F, "N"))
    out["nature"] = "repulsiva" if F > 0 else "atrativa" if F < 0 else "nula"
    return out


def electric_field(charge, distance):
    """Campo elétrico de carga pontual: E = k q / r²."""
    q, r = _n(charge, "charge"), _p(distance, "distance")
    return _m(r"E=\frac{kq}{r^2}", field=(K * q / r**2, "N/C"), potential=(K * q / r, "V"))


def ohm(voltage=None, current=None, resistance=None):
    """Lei de Ohm V = R I; deixe um vazio."""
    miss = _one_missing(voltage=voltage, current=current, resistance=resistance)
    V = _n(voltage, "voltage") if voltage is not None else None
    I = _n(current, "current") if current is not None else None
    Rr = _p(resistance, "resistance") if resistance is not None else None
    if miss == "voltage":
        V = Rr * I
    elif miss == "current":
        I = V / Rr
    else:
        if I == 0:
            raise MathError("Corrente nula: resistência indeterminada.")
        Rr = V / I
        if Rr <= 0:
            raise MathError("Resistência resultante não positiva.")
    return _m(r"V=RI", voltage=(V, "V"), current=(I, "A"), resistance=(Rr, "Ω"), power=(V * I, "W"))


def _combine(values, mode, name):
    if not isinstance(values, list) or not values:
        raise MathError(f"{name}: informe uma lista não vazia.")
    vs = [_p(x, name) for x in values]
    if mode not in ("series", "parallel"):
        raise MathError("mode deve ser 'series' ou 'parallel'.")
    return vs


def resistors(values, mode="series"):
    """Associação de resistores (Ω): série ou paralelo."""
    vs = _combine(values, mode, "resistor")
    eq = sum(vs) if mode == "series" else 1 / sum(1 / x for x in vs)
    return _m(r"R_{eq}=\sum R_i" if mode == "series" else r"\frac1{R_{eq}}=\sum\frac1{R_i}", equivalent=(eq, "Ω"))


def capacitors(values, mode="parallel"):
    """Associação de capacitores (F): série ou paralelo."""
    vs = _combine(values, mode, "capacitor")
    eq = sum(vs) if mode == "parallel" else 1 / sum(1 / x for x in vs)
    return _m(r"C_{eq}=\sum C_i" if mode == "parallel" else r"\frac1{C_{eq}}=\sum\frac1{C_i}", equivalent=(eq, "F"))


def magnetic_field_wire(current, distance):
    """Campo de fio retilíneo longo: B = μ0 I / (2π r)."""
    return _m(r"B=\frac{\mu_0I}{2\pi r}", field=(MU0 * _n(current, "current") / (2 * math.pi * _p(distance, "distance")), "T"))


def faraday_emf(turns, delta_flux, delta_t):
    """Lei de Faraday: ε = −N ΔΦ/Δt."""
    return _m(r"\varepsilon=-N\frac{\Delta\Phi}{\Delta t}", emf=(-_p(turns, "turns") * _n(delta_flux, "delta_flux") / _p(delta_t, "delta_t"), "V"))


# ---- ondas e óptica ----
def wave(speed=None, frequency=None, wavelength=None):
    """Equação fundamental da onda v = f λ; deixe um vazio."""
    d = {"speed": speed, "frequency": frequency, "wavelength": wavelength}
    miss = _one_missing(**d)
    v = {k: _p(x, k) for k, x in d.items() if x is not None}
    calc = {"speed": lambda: v["frequency"] * v["wavelength"], "frequency": lambda: v["speed"] / v["wavelength"],
            "wavelength": lambda: v["speed"] / v["frequency"]}
    return _m(r"v=f\lambda", **{miss: (calc[miss](), {"speed": "m/s", "frequency": "Hz", "wavelength": "m"}[miss])})


def snell(n1, n2, theta1_deg):
    """Lei de Snell: n1 sinθ1 = n2 sinθ2 (detecta reflexão total interna)."""
    n1, n2, t1 = _p(n1, "n1"), _p(n2, "n2"), _n(theta1_deg, "theta1_deg")
    if not 0 <= t1 < 90:
        raise MathError("theta1_deg deve estar em [0, 90).")
    s = n1 * math.sin(math.radians(t1)) / n2
    if s > 1:
        raise MathError("Reflexão total interna: não há raio refratado.")
    return _m(r"n_1\sin\theta_1=n_2\sin\theta_2", theta2=(math.degrees(math.asin(s)), "°"))


def _imaging(f, do, note):
    f, do = _n(f, "focal_length"), _p(do, "object_distance")
    if f == 0:
        raise MathError("Distância focal não pode ser zero.")
    inv = 1 / f - 1 / do
    if abs(inv) < 1e-12:
        raise MathError("Objeto no foco: imagem no infinito.")
    di = 1 / inv
    out = _m(r"\frac1f=\frac1{d_o}+\frac1{d_i},\ m=-\frac{d_i}{d_o}", image_distance=(di, "m"),
             magnification=(-di / do, "adimensional"))
    out["image"] = "real" if di > 0 else "virtual"
    out["note"] = note
    return out


def thin_lens(focal_length, object_distance):
    """Lente delgada (f>0 convergente, f<0 divergente)."""
    return _imaging(focal_length, object_distance, "Convenção: di>0 imagem real.")


def spherical_mirror(focal_length, object_distance):
    """Espelho esférico (f>0 côncavo, f<0 convexo)."""
    return _imaging(focal_length, object_distance, "Convenção: di>0 imagem real, na frente do espelho.")


def double_slit(order, wavelength, slit_distance, screen_distance):
    """Interferência (fenda dupla), máximos: y ≈ m λ L / d (ângulos pequenos)."""
    m = _n(order, "order")
    y = m * _p(wavelength, "wavelength") * _p(screen_distance, "screen_distance") / _p(slit_distance, "slit_distance")
    out = _m(r"y\approx\frac{m\lambda L}{d}", position=(y, "m"))
    out["note"] = "Aproximação de ângulos pequenos."
    return out


METHODS = {f.__name__: f for f in (
    kinematics_mruv, torricelli, circular_motion, newton_second, weight, work, kinetic_energy, potential_energy,
    collision_1d, gravitation, torque, moment_of_inertia, spring_oscillator, pendulum_period, ideal_gas,
    thermal_expansion, heat, carnot_efficiency, entropy_change, convert_temperature, coulomb, electric_field,
    ohm, resistors, capacitors, magnetic_field_wire, faraday_emf, wave, snell, thin_lens, spherical_mirror,
    double_slit)}


def run(method, params):
    if method not in METHODS:
        raise MathError(f"Método desconhecido: {method}.")
    try:
        return METHODS[method](**(params or {}))
    except TypeError as exc:
        raise MathError(f"Parâmetros inválidos para {method}: {exc}")
