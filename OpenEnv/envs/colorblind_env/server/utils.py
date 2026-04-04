from colorspacious import cspace_convert
from skimage.color import deltaE_ciede2000

# Takes a color value (using hex code) and return a color value (hex code)
# for the particular Color Blind Person.
def simulate_cb(hex_code, cb_type, severity=100):
    '''
    Takes 
    hex_code : "#FF0000",
    cb_type : "protanopia",
    severity : 100

    Return a Hex code of the color that the Color blind person will see

    Note : By default the colorspacious, use the anamoly convention rather than opia, so 
    to simulate *opia, we have to use the severity value, which for now I have set it to 100.
    '''

    opia_to_anomaly_convert = {
        "protanopia": "protanomaly",
        "deuteranopia": "deuteranomaly",
        "tritanopia": "tritanomaly",
    }

    cb_type_str = cb_type.value if hasattr(cb_type, 'value') else cb_type

    cvd_space = {
        "name" : "sRGB1+CVD",
        "cvd_type": opia_to_anomaly_convert[cb_type_str],
        "severity": severity,
    }

    r, g, b = hex_to_rgb(hex_code)

    rgb_norm = [r / 255, g / 255, b / 255]

    # simulate color blindness
    simulated = cspace_convert(rgb_norm, "sRGB1", cvd_space)

    # convert back to 0-255
    simulated_255 = [int(max(0, min(1, c)) * 255) for c in simulated]

    simulated_hexcode = rgb_to_hex(*simulated_255)

    return simulated_hexcode

# 
def compute_delta_e(hex1, hex2, cb_type):
    simulated_hex1 = simulate_cb(hex1, cb_type)
    simulated_hex2 = simulate_cb(hex2, cb_type)

    simulated_rgb1 = hex_to_rgb(simulated_hex1)
    simulated_rgb2 = hex_to_rgb(simulated_hex2)

    simulated_lab1 = rgb_to_lab(*simulated_rgb1)
    simulated_lab2 = rgb_to_lab(*simulated_rgb2)

    color_dist = deltaE_ciede2000(simulated_lab1, simulated_lab2)

    return color_dist

# Hex Value to RGB Value
def hex_to_rgb(hex_code):
    hex_code = hex_code.lstrip('#')  # remove '#'
    return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))

# RGB value to HEX Value
def rgb_to_hex(r, g, b):
    return "#{:02X}{:02X}{:02X}".format(r, g, b)

# RGB value to LAB value
def rgb_to_lab(r, g, b):
    r_norm = r / 255
    g_norm = g / 255
    b_norm = b / 255
    
    rgb_norm = [r_norm, g_norm, b_norm]

    lab = cspace_convert(rgb_norm, "sRGB1", "CIELab")

    return lab

# LAB value to RGB
def lab_to_rgb(lab):
    rgb_back = cspace_convert(lab, "CIELab", "sRGB1")

    # scale back to [0,255]
    # rgb_back = [int(x * 255) for x in rgb_back]
    rgb_back = [int(max(0, min(1, x)) * 255) for x in rgb_back]

    return rgb_back
