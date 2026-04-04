import matplotlib.pyplot as plt
import numpy as np
from server.utils import simulate_cb_hex as simulate_cb

def plot_categories(categories, title="Plot"):
    fig = plt.figure(figsize=(6, 6))

    for name, cat in categories.items():
        points = np.array(cat.points)
        color = cat.hex
        shape = cat.shape

        plt.scatter(points[:, 0], points[:, 1],
                    c=color,
                    marker=shape,
                    label=name,
                    s=50)

    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    return fig


def simulate_plot(categories, cb_type):
    simulated_categories = {}

    for name, cat in categories.items():
        sim_hex = simulate_cb(cat.hex, cb_type)

        # DEBUG PRINT (add this once)
        print(f"{name}: {cat.hex} → {sim_hex}")

        simulated_categories[name] = type(cat)(
            hex=sim_hex,
            shape=cat.shape,
            points=cat.points
        )

    return simulated_categories