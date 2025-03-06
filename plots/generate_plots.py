import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import numpy as np
import os

result_dirs = ["Feb01_1309", "Feb01_0354", "Feb01_0346"]
result_titles = [r"$K = 5, N = 5, d = 5, L_0 = 0.5$",
                 r"$K = 5, N = 100, d = 10, L_0 = 0.5$",
                 r"$K = 10, N = 100, d = 10, L_0 = 0.1$"]

plt.style.use(['paper.mplstyle'])
matplotlib.rcParams.update({"axes.grid": False})
rc('font', **{'family': 'serif', 'serif': ['Nimbus Roman No9 L']})

method_names = ["CAP (Algorithm 2)", "CAP-ONS (Algorithm 3)", "M3P (Javanmard et al., 2020)", "DBL-MNL (Oh & Iyengar, 2021)", "DBL-MNL (Oh & Iyengar, 2021) + Dynamic Pricing", "ONS-MPP (Perivier & Goyal, 2022)"]

len_methods = len(method_names)

result_y_lims = [1000, 2000, 6000]

def plot_results(ax, data_collection, plt_i):
    data_log = np.stack(data_collection, axis=-1)
    optimum_revenues = data_log[0]

    temp = data_log[5].copy()
    data_log[5] = data_log[2].copy()
    data_log[2] = temp

    list_of_regrets = []
    list_of_cum_regrets = []

    T = len(data_log[0])

    for m in range(len(method_names)):
        regret = np.maximum(optimum_revenues - data_log[m+1], 0)
        list_of_regrets.append(regret)
        list_of_cum_regrets.append(np.cumsum(list_of_regrets[m], axis=0))

    # generate_plots(ax, [dynamic_regret, newton_regret], "Regret", "Regret")
    generate_plots(ax, list_of_cum_regrets, "Cumulative Regret", plt_i)


def generate_plots(ax, data_log, y_label, plt_i):
    num_of_methods = len(data_log)
    T, S = data_log[0].shape

    error_every = 400
    for m in range(num_of_methods):
        mean_data =np.mean(data_log[m], axis=-1)
        std_data = np.std(data_log[m], axis=-1)
        markers, caps, bars = ax.errorbar(np.linspace(1, T + 1, T), mean_data, yerr=np.minimum(std_data, 100),
                                          errorevery=(100 + (error_every//len_methods) * (m + 1), error_every), capsize=2)
        [bar.set_alpha(0.5) for bar in bars]
        [cap.set_alpha(0.5) for cap in caps]

    ax.set(xlabel='Iteration', ylabel=y_label)
    ax.grid()
    ax.set_ylim(bottom=1e-4, top=result_y_lims[plt_i])
    ax.set_xlim(left=1e-4)
    ax.set_xticks(np.arange(0, T + 1, 400))
    ax.set_title(result_titles[plt_i], x=0.48, y=1.0, pad=4)


fig, ax = plt.subplots(1, max(len(result_dirs), 2), figsize=(6, 1.9), dpi=400)
fig.tight_layout(w_pad=0.3, h_pad=0.2, rect=(0.01, 0.01, 0.998, 0.78))
fig.subplots_adjust(wspace=0.3, hspace=0.1)


exp_count = 0
for i, result_dir in enumerate(result_dirs):
    data_collection = []
    for filename in os.listdir(f"results/{result_dir}"):
        try:
            loaded_data = np.load(f"results/{result_dir}/{filename}/history_expected_revenue.npy")
            data_collection.append(loaded_data)
            exp_count += 1
        except NotADirectoryError:
            continue
    plot_results(ax[i], data_collection, i)


fig.legend(method_names, loc='upper center', frameon=True, ncol=len(method_names)//2, columnspacing=1, prop=dict(size=7))
plt.savefig(f"exp_main.pdf")
plt.show()