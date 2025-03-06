import numpy as np
import matplotlib.pyplot as plt
import seaborn
from matplotlib import rc
from matplotlib import rcParams

import matplotlib
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import pyplot


matplotlib.rcParams.update({"axes.grid": False})
rc('font', **{'family': 'serif', 'serif': ['Nimbus Roman No9 L']})

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Computer Model Roman",
    'font.weight': 'bold'
})

# plt.rc('text', usetex=True)
rcParams['text.latex.preamble'] = r'\usepackage{bm}'

seaborn.set(style='ticks')

a_true = 0.73
b_true = 0.15

a_center = 0.8
b_center = 0.2

norm = mpl.colors.Normalize(vmin=0, vmax=np.pi)
cmap = cm.get_cmap("winter")

m = cm.ScalarMappable(norm=norm, cmap=cmap)

def u_fun(p, a, b):
    return a - p * b

P_range = np.linspace(-.1, 4.1, 100)

r_1 = 0.21
r_2 = 0.06
phi = np.pi / 6
# phi = 0

fig, ax = plt.subplots(figsize=(6, 3.5), dpi=500)
fig.subplots_adjust(bottom=0.19, top=0.92, left=0.13, right=0.9)

inset_axes = inset_axes(ax,
                    width=1.8,                     # inch
                    height=1.8,                    # inch
                    bbox_transform=ax.transAxes, # relative axes coordinates
                    bbox_to_anchor=(0.55, 0.5),    # relative axes coordinates
                    loc=3)                       # loc=lower left corner

ax.axis([0, max(P_range), 0, 1])

# fig.subplots_adjust(bottom=0.2, top=0.95, left=0.15, right=0.95)

for theta in np.linspace(0, 2 * np.pi, 1000):
    z = np.exp(theta*1j)
    z = r_1 * z.real + r_2 * z.imag * 1j
    z = z * np.exp(phi*1j)
    a_gap = np.real(z)
    b_gap = np.imag(z)
    if theta < np.pi:
        color_val = theta
    else:
        color_val = 2 * np.pi - theta
    ax.plot(P_range, u_fun(P_range, a_center + a_gap, b_center + b_gap), color=m.to_rgba(color_val), alpha=0.01)

ax.plot(P_range, u_fun(P_range, a_true, b_true), color="#d11141", linewidth=2.5)
ax.text(2.97, 0.1, r"$u_{ti}(p)$", color='#d11141', fontsize=23)

def h_fun(p):
    c_1 = r_1**2 * (np.cos(phi)) ** 2 + r_2 ** 2 * (np.sin(phi)) ** 2
    c_2 = - 2 * (r_1**2 - r_2**2) * np.cos(phi) * np.sin(phi)
    c_3 = r_1**2 * (np.sin(phi)) ** 2 + r_2 ** 2 * (np.cos(phi)) ** 2
    return a_center - b_center * p + np.sqrt(c_1 + c_2 * p + c_3 * p ** 2) + 0.01


ax.plot(P_range, h_fun(P_range), color="#FF7500", linewidth=2.5, linestyle="--")
ax.text(2.97, 0.48, r"$h_{ti}(p)$", color='#FF7500', fontsize=23)

ax.grid(True, which='both')
# seaborn.despine(ax=ax)

ax.spines[['left', 'bottom']].set_position('zero')
ax.spines[['top', 'right']].set_visible(False)

ax.set_xlabel(r"$p$ (price)", fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=15)

def color_function(a, b):
    z = (a - a_center) + (b - b_center) * 1j
    z = z * np.exp(phi * 1j)
    h = (z.real / r_1) + (z.imag / r_2) * 1j
    angles_pos = np.angle(h) >= 0
    angles = angles_pos * np.angle(h) + (1 - angles_pos) * (- np.angle(h))
    return angles * (np.abs(h) <= 1)

# make these smaller to increase the resolution
dx, dy = 0.0001, 0.0001

x = np.arange(0.55, 1.05, dx)
y = np.arange(0.05, 0.35, dy)
X, Y = np.meshgrid(x, y)

xmin, xmax, ymin, ymax = np.amin(x), np.amax(x), np.amin(y), np.amax(y)
extent = xmin, xmax, ymin, ymax

colors = color_function(X, Y)

print(colors)

cmap.set_under('w')

inset_axes.imshow(colors, cmap=cmap, alpha=0.6, extent=extent, vmin=0.001)
inset_axes.scatter(a_true, b_true, color='#d11141', marker="*", s=50)
inset_axes.scatter(a_center, b_center, color='#ffc425', marker="^", s=50)
inset_axes.text(0.68, 0.15, r"$\boldsymbol{\theta}^*$", color='#d11141', fontsize=15)
inset_axes.text(0.82, 0.20, r"$\widehat{\boldsymbol{\theta}}_t$", color='#ffc425', fontsize=15)
inset_axes.set_xticks([])
inset_axes.set_yticks([])

plt.savefig(f"utility_function.pdf")
plt.show()

