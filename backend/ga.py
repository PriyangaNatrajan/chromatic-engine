# backend/ga.py
import numpy as np
import random
import math
import colorsys

# ============================================================
# COLOR CONVERSIONS
# ============================================================

def rgb_to_xyz(rgb):
    r, g, b = rgb
    def lin(u):
        return ((u + 0.055) / 1.055) ** 2.4 if u > 0.04045 else u / 12.92
    r, g, b = lin(r), lin(g), lin(b)

    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return M.dot(np.array([r, g, b]))


def xyz_to_lab(xyz):
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x, y, z = xyz[0] / Xn, xyz[1] / Yn, xyz[2] / Zn

    def f(t):
        delta = 6/29
        return t ** (1/3) if t > delta**3 else (t / (3 * delta**2) + 4/29)

    fx, fy, fz = f(x), f(y), f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.array([L, a, b])


def rgb_to_lab(rgb):
    return xyz_to_lab(rgb_to_xyz(rgb))


def lab_to_xyz(lab):
    L, a, b = lab
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def finv(t):
        delta = 6/29
        return t**3 if t > delta else 3 * delta**2 * (t - 4/29)

    x = finv(fx)
    y = finv(fy)
    z = finv(fz)

    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    return np.array([x * Xn, y * Yn, z * Zn])


def xyz_to_rgb(xyz):
    M = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    r_lin, g_lin, b_lin = M.dot(xyz)

    def comp(u):
        return 1.055 * (u ** (1/2.4)) - 0.055 if u > 0.0031308 else 12.92 * u

    r, g, b = comp(r_lin), comp(g_lin), comp(b_lin)
    return np.clip([r, g, b], 0.0, 1.0)


def lab_to_rgb(lab):
    return xyz_to_rgb(lab_to_xyz(lab))


def hex_to_rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i+2], 16)/255.0 for i in (0,2,4)], dtype=float)


def rgb_to_hex(rgb):
    rgb = np.clip(rgb, 0, 1)
    r, g, b = (rgb * 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

# ============================================================
# CIEDE2000
# ============================================================

def cie2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    Lp = 0.5 * (L1 + L2)
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    Cp = 0.5 * (C1 + C2)

    G = 0.5 * (1 - math.sqrt((Cp**7) / (Cp**7 + 25**7)))
    a1p = (1 + G) * a1
    a2p = (1 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)
    hp1 = math.degrees(math.atan2(b1, a1p)) % 360
    hp2 = math.degrees(math.atan2(b2, a2p)) % 360

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = hp2 - hp1
    if C1p * C2p != 0:
        if abs(dhp) > 180:
            dhp += 360 if hp2 <= hp1 else -360
    else:
        dhp = 0

    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    Lpm = (L1 + L2) / 2.0
    Cpm = (C1p + C2p) / 2.0

    if C1p * C2p == 0:
        hpmp = hp1 + hp2
    else:
        hh = abs(hp1 - hp2)
        hpmp = (hp1 + hp2 + 360) / 2.0 if hh > 180 else (hp1 + hp2) / 2.0

    T = (
        1
        - 0.17 * math.cos(math.radians(hpmp - 30))
        + 0.24 * math.cos(math.radians(2 * hpmp))
        + 0.32 * math.cos(math.radians(3 * hpmp + 6))
        - 0.20 * math.cos(math.radians(4 * hpmp - 63))
    )

    delta_ro = 30 * math.exp(-((hpmp - 275) / 25.0) ** 2)
    RC = 2 * math.sqrt((Cpm**7) / (Cpm**7 + 25**7))
    SL = 1 + (0.015 * ((Lpm - 50) ** 2)) / math.sqrt(20 + ((Lpm - 50) ** 2))
    SC = 1 + 0.045 * Cpm
    SH = 1 + 0.015 * Cpm * T
    RT = -math.sin(math.radians(2 * delta_ro)) * RC

    dE = math.sqrt(
        (dLp / SL) ** 2 +
        (dCp / SC) ** 2 +
        (dHp / SH) ** 2 +
        RT * (dCp / SC) * (dHp / SH)
    )
    return dE

# ============================================================
# MODE HELPERS
# ============================================================

def rgb_to_hsv_deg(rgb):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return h * 360.0, s, v


def chroma_from_lab(lab):
    return math.hypot(lab[1], lab[2])


def mode_enforce(rgb, mode):
    if mode == "pastel":
        lab = rgb_to_lab(rgb)
        lab[0] = max(lab[0], 80.0)
        lab[1] *= 0.45
        lab[2] *= 0.45
        return lab_to_rgb(lab)

    if mode == "vibrant":
        h, s, v = rgb_to_hsv_deg(rgb)
        return np.array(colorsys.hsv_to_rgb(h/360, min(1, s*1.6), min(1, v*1.05)))

    if mode == "neon":
        h, s, v = rgb_to_hsv_deg(rgb)
        out = np.array(colorsys.hsv_to_rgb(h/360, min(1, s*1.8), max(0.85, v)))
        return np.clip(out**0.8, 0, 1)

    if mode == "earthy":
        h, s, v = rgb_to_hsv_deg(rgb)
        h = h + 0.1 * ((35 - h + 540) % 360 - 180)
        s = max(0.2, s * 0.9)
        v = max(0.15, v * 0.9)
        return np.array(colorsys.hsv_to_rgb(h/360, s, v))

    if mode == "mono":
        lab = rgb_to_lab(rgb)
        lab[1] *= 0.05
        lab[2] *= 0.05
        return lab_to_rgb(lab)

    if mode == "gradient":
        lab = rgb_to_lab(rgb)
        lab[0] = lab[0] * 0.95 + 10
        return lab_to_rgb(lab)

    return rgb

# ============================================================
# GENETIC ALGORITHM CLASS
# ============================================================

class PaletteGA:
    def __init__(
        self,
        n_colors=5,
        pop_size=120,
        elite_size=6,
        mutation_rate=0.18,
        generations=60,
        seed=None,
        mode="default",
        base_color=None
    ):
        if seed:
            np.random.seed(seed)
            random.seed(seed)

        self.n_colors = n_colors
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.mode = mode
        self.base_color = base_color

        self.base_lab = rgb_to_lab(hex_to_rgb(base_color)) if base_color else None

        self.population = self.init_population()


    def init_population(self):
        pop = []
        for _ in range(self.pop_size):
            palette = np.random.rand(self.n_colors, 3)
            palette = np.array([mode_enforce(c, self.mode) for c in palette])
            if self.base_color:
                palette[0] = hex_to_rgb(self.base_color)
            pop.append(palette)
        return pop


    def mutate(self, palette):
        child = palette.copy()
        for i in range(self.n_colors):
            if random.random() < self.mutation_rate:
                noise = np.random.normal(scale=0.10, size=3)
                new = np.clip(child[i] + noise, 0, 1)
                child[i] = mode_enforce(new, self.mode)

        if self.base_color:
            child[0] = hex_to_rgb(self.base_color)
        return child


    def crossover(self, p1, p2):
        cut = random.randint(1, self.n_colors - 1)
        child = np.vstack((p1[:cut], p2[cut:]))
        child = np.array([mode_enforce(c, self.mode) for c in child])
        if self.base_color:
            child[0] = hex_to_rgb(self.base_color)
        return child


    def tournament_select(self, scored, k=3):
        chosen = random.sample(scored, k)
        chosen.sort(key=lambda x: x[1], reverse=True)
        return chosen[0][0]


    def fitness(self, palette):
        labs = [rgb_to_lab(c) for c in palette]
        pair_dE = [cie2000(labs[i], labs[j]) for i in range(len(labs)) for j in range(i+1, len(labs))]
        mean_dE = np.mean(pair_dE)

        score = np.clip(mean_dE / 45.0, 0, 1)
        score += random.uniform(-0.02, 0.02)
        return float(score)


    def evolve(self):
        for _ in range(self.generations):
            scored = [(p, self.fitness(p)) for p in self.population]
            scored.sort(key=lambda x: x[1], reverse=True)
            elites = [p for p, _ in scored[:self.elite_size]]

            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_select(scored)
                p2 = self.tournament_select(scored)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)

            self.population = new_pop


    def get_top_palettes_hex(self, top_k=6):
        self.evolve()
        scored = [(p, self.fitness(p)) for p in self.population]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for pal, score in scored[:top_k]:
            hexes = [rgb_to_hex(c) for c in pal]
            results.append({
                "hex": hexes,
                "score": float(score)
            })
        return results


    def evolve_with_history(self):
        """Run full GA + return fitness curve + best palette per generation."""
        self.population = self.init_population()  # reset every time user presses “Show Convergence”

        history = []

        for gen in range(self.generations):
            scored = [(p, self.fitness(p)) for p in self.population]
            fitness_vals = [s for _, s in scored]

            best_idx = int(np.argmax(fitness_vals))

            history.append({
                "generation": gen,
                "max_fitness": float(np.max(fitness_vals)),
                "mean_fitness": float(np.mean(fitness_vals)),
                "min_fitness": float(np.min(fitness_vals)),
                "best_palette": [rgb_to_hex(c) for c in self.population[best_idx]]
            })

            scored.sort(key=lambda x: x[1], reverse=True)
            elites = [p for p, _ in scored[:self.elite_size]]

            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1 = self.tournament_select(scored)
                p2 = self.tournament_select(scored)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_pop.append(child)

            self.population = new_pop

        return history 
