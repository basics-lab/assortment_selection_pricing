import numpy as np
import matplotlib.pyplot as plt
import seaborn
from matplotlib import rc
import matplotlib

matplotlib.rcParams.update({"axes.grid": False})
rc('font', **{'family': 'serif', 'serif': ['Nimbus Roman No9 L']})

seaborn.set(style='ticks')

a_true = 0.8
b_true = 0.2

def u_fun(p, a, b):
    return a - p * b

P_range = np.linspace(0, 3, 100)

r_1 = 0.15
r_2 = 0.03
phi = np.pi / 4

fig, ax = plt.subplots(figsize=(5, 3))
fig.subplots_adjust(bottom=0.2, top=0.95, left=0.15, right=0.95)

for theta in np.linspace(0, 2 * np.pi, 1000):
    z = np.exp(theta*1j)
    z = r_1 * z.real + r_2 * z.imag * 1j
    z = z * np.exp(phi*1j)
    a_gap = np.real(z)
    b_gap = np.imag(z)
    ax.plot(P_range, u_fun(P_range, a_true + a_gap, b_true + b_gap), "blue", alpha=0.01)

ax.plot(P_range, u_fun(P_range, a_true, b_true), "red", linewidth=2.5)
ax.set_xlabel(r"$p$")
ax.set_ylabel(r"$u(p)$")

ax.grid(True, which='both')
seaborn.despine(ax=ax)

plt.show()
