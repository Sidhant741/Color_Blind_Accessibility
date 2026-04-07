"""
Color utilities: simulation + perceptual distance
"""

from colorspacious import cspace_convert
from skimage.color import deltaE_ciede2000


def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def rgb_to_lab(r, g, b):
    return cspace_convert([r/255, g/255, b/255], "sRGB1", "CIELab")


def simulate_cb(hex_code, cb_type):
    opia_map = {
        "protanopia": "protanomaly",
        "deuteranopia": "deuteranomaly",
        "tritanopia": "tritanomaly",
    }

    r, g, b = hex_to_rgb(hex_code)

    sim = cspace_convert(
        [r/255, g/255, b/255],
        "sRGB1",
        {
            "name": "sRGB1+CVD",
            "cvd_type": opia_map[cb_type],
            "severity": 100,
        }
    )

    sim = [int(max(0, min(1, c)) * 255) for c in sim]

    return rgb_to_hex(*sim)


def compute_delta_e(hex1, hex2, cb_type):
    sim1 = simulate_cb(hex1, cb_type)
    sim2 = simulate_cb(hex2, cb_type)

    lab1 = rgb_to_lab(*hex_to_rgb(sim1))
    lab2 = rgb_to_lab(*hex_to_rgb(sim2))

    return deltaE_ciede2000(lab1, lab2)


# 🔥 MULTI-CB DISTANCE
def compute_multi_cb_delta(hex1, hex2, cb_types):
    deltas = [compute_delta_e(hex1, hex2, cb) for cb in cb_types]
    return min(deltas)  # worst-case