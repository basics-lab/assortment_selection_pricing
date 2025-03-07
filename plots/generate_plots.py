import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
import numpy as np
import glob
import pandas as pd
import seaborn as sns

result_dirs = ["results/20250306_104954", "results/20250306_104959", "results/20250306_105005"]
result_titles = [r"$K = 5, N = 5, d = 5, L_0 = 0.5$",
                 r"$K = 5, N = 100, d = 10, L_0 = 0.5$",
                 r"$K = 10, N = 100, d = 10, L_0 = 0.1$"]

plt.style.use(['paper.mplstyle'])
matplotlib.rcParams.update({"axes.grid": False})
# rc('font', **{'family': 'serif', 'serif': ['Nimbus Roman No9 L']})
rc('text', usetex=True)

method_names = {
    "CAP": r"\textbf{CAP (Algorithm 2)}",
    "CAP-ONS": r"\textbf{CAP-ONS (Algorithm 3)}",
    "M3P": r"M3P (Javanmard et al., 2020)",
    "ONS-MPP": r"ONS-MPP (Perivier \& Goyal, 2022)",
    "DBL-MNL-Pricing": r"DBL-MNL (Oh \& Iyengar, 2021) \textbf{+ CAP Pricing}",
    "Thompson": r"TS-MNL (Oh \& Iyengar, 2019) \textbf{+ CAP Pricing}"
}

colors = dict(zip(method_names.keys(), sns.color_palette("tab10")))

rename = {
    "CAP": "CAP-ONS",
    "CAP-ONS": "DBL-MNL-Pricing",
    "DBL-MNL-Pricing": "CAP"
}

len_methods = len(method_names)
y_label = r"Cumulative Regret"

result_y_lims = [380, 1600, 4800]
T = 2000

def plot_results(ax, results_df, plt_i):
    results_df = results_df[results_df['T'] == T].copy()

    results_by_algo = results_df.groupby('algo')
    for m, (algo_name, algo_results) in enumerate(results_by_algo):
        if algo_name in rename:
            algo_name = rename[algo_name]
        if algo_name in method_names:
            algo_results['regret'] = algo_results['optimal_revenue'] - algo_results['expected_revenue']
            algo_results['cumulative_regret'] = algo_results.groupby('run_id')['regret'].cumsum()
            algo_average = algo_results[['cumulative_regret', 't']].groupby('t').mean().reset_index()
            algo_std = algo_results[['cumulative_regret', 't']].groupby('t').std().reset_index()
            t = algo_average['t']
            avg_regret = algo_average['cumulative_regret']
            std_regret = 0.5 * algo_std['cumulative_regret']
            ax.plot(t, avg_regret, label=method_names[algo_name], color=colors[algo_name])
            ax.fill_between(t, avg_regret - std_regret, avg_regret + std_regret, color=colors[algo_name], alpha=0.1)

    ax.set(xlabel=r'Iteration', ylabel=y_label)
    ax.grid()
    ax.set_ylim(bottom=1e-4, top=result_y_lims[plt_i])
    ax.set_xlim(left=1e-4)
    ax.set_xticks(np.arange(0, T + 1, 400))
    ax.set_title(result_titles[plt_i], x=0.48, y=1.0, pad=4)


fig, ax = plt.subplots(1, max(len(result_dirs), 2), figsize=(6.4, 1.9), dpi=400)
fig.tight_layout(w_pad=0.3, h_pad=0.2, rect=(0.01, 0.01, 0.998, 0.78))
fig.subplots_adjust(wspace=0.3, hspace=0.1)


exp_count = 0
for i, result_folder in enumerate(result_dirs):
    result_files = glob.glob(f"{result_folder}/*/results.parquet")
    run_dfs = [pd.read_parquet(f) for f in result_files]
    for run_id, df in enumerate(run_dfs):
        df['run_id'] = run_id
    results_df = pd.concat(run_dfs, ignore_index=True)
    plot_results(ax[i], results_df, i)

handles = [plt.Line2D([0], [0], color=colors[method], lw=1) for method in method_names.keys()]
labels = [method_names[method] for method in method_names.keys()]
fig.legend(handles, labels, loc='upper center', frameon=True, ncol=3, columnspacing=1, prop=dict(size=8))
plt.savefig(f"plots/exp_main.pdf")
plt.show()