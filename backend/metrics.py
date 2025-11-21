import numpy as np
import colorsys

# -------------------------------------------------
# RGB → XYZ → LAB (Pure NumPy, no scikit-image)
# -------------------------------------------------

def rgb_to_xyz(rgb):
    rgb = np.array(rgb)
    mask = rgb > 0.04045
    rgb_lin = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)

    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    return np.dot(M, rgb_lin)


def xyz_to_lab(xyz):
    XYZ_ref = np.array([0.95047, 1.00000, 1.08883])  # D65 reference white
    xyz_scaled = xyz / XYZ_ref

    def f(t):
        delta = 6/29
        return np.where(
            t > delta**3,
            np.cbrt(t),
            t / (3 * delta**2) + 4/29
        )

    fx, fy, fz = f(xyz_scaled)
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.array([L, a, b])


def rgb_to_lab(rgb):
    return xyz_to_lab(rgb_to_xyz(rgb))


# -------------------------------------------------
# Utility: RGB → HEX
# -------------------------------------------------

def rgb_to_hex(rgb):
    r, g, b = (np.clip(rgb, 0, 1) * 255).astype(int)
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


# -------------------------------------------------
# Contrast (WCAG)
# -------------------------------------------------

def linearize_channel(c):
    return np.where(
        c <= 0.03928,
        c / 12.92,
        ((c + 0.055) / 1.055) ** 2.4
    )


def relative_luminance(rgb):
    r, g, b = rgb
    r_lin, g_lin, b_lin = linearize_channel(np.array([r, g, b]))
    return 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin


def contrast_ratio(rgb1, rgb2):
    L1 = relative_luminance(rgb1)
    L2 = relative_luminance(rgb2)
    lighter = max(L1, L2)
    darker = min(L1, L2)
    return (lighter + 0.05) / (darker + 0.05)


# -------------------------------------------------
# LAB Distance → Color Diversity
# -------------------------------------------------

def mean_lab_distance(palette_rgb):
    labs = np.array([rgb_to_lab(rgb) for rgb in palette_rgb])
    dists = []

    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            d = np.linalg.norm(labs[i] - labs[j])
            dists.append(d)

    return float(np.mean(dists)) if dists else 0.0


# -------------------------------------------------
# Harmony Score based on Hue Relationships
# -------------------------------------------------

def rgb_to_hue(rgb):
    h, l, s = colorsys.rgb_to_hls(*rgb)
    return h * 360.0


def hue_distance(a, b):
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def harmony_score(palette_rgb):
    hues = [rgb_to_hue(rgb) for rgb in palette_rgb]

    score = 0.0
    pairs = 0

    for i in range(len(hues)):
        for j in range(i + 1, len(hues)):
            d = hue_distance(hues[i], hues[j])
            pairs += 1

            if d <= 30:
                score += 1.0  # analogous
            elif abs(d - 120) <= 15:
                score += 1.2  # triadic
            elif abs(d - 180) <= 20:
                score += 1.5  # complementary
            else:
                score += max(0.0, 0.4 * (1 - (d / 180.0)))

    if pairs == 0:
        return 0.0

    return float(score / pairs) 