# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: torch-rocm
#     language: python
#     name: python3
# ---

# %%
from utilities.data_loader import load_data

train_prices, train_dates, test_prices, test_dates = load_data("./data/BTC-Daily.csv")

# %% [markdown]
# We are going to determine a naive solution of Buy and Hold as benchmark for performance later

# %%
from optimizer.evaluator import execute_buy, execute_sell
from math import ceil
_, btc_held = execute_buy(1000, test_prices[0], 0.03)
BUY_AND_HOLD_BENCHMARK, _ = execute_sell(btc_held, test_prices[-1], 0.03)
BUY_AND_HOLD_BENCHMARK = ceil(BUY_AND_HOLD_BENCHMARK)


# %%
import pandas as pd

fixed_results = pd.read_csv("results/fixed_runs.csv")
random_results = pd.read_csv("results/random_runs.csv")

# %% [markdown]
# Statistical Summary of test final cash on fixed and random point

# %%
df = pd.concat([fixed_results, random_results], ignore_index=True)

# Flip fitness because gps is maximizing not minimizing
df.loc[df["algo"] == "gps", "best_fitness"] *= -1

def fmt(x):
    if pd.isna(x):
        return "NaN"
    if abs(x) < 1e-7 or abs(x) > 9_999_999:
        return f"{x:.3e}"
    if x == 0:
        return "0"
    return f"{x:,.4f}"

stats = (
    df.groupby(['start_mode', 'algo', 'strategy'])['test_final_cash']
    .agg(count='count', mean='mean', std='std', var='var', min='min', median='median', max='max')
)
display(stats.map(fmt))

# %% [markdown]
# On the training set

# %%
df = pd.concat([fixed_results, random_results], ignore_index=True)

# Flip fitness because gps is maximizing not minimizing
df.loc[df["algo"] == "gps", "best_fitness"] *= -1

def fmt(x):
    if pd.isna(x):
        return "NaN"
    if abs(x) < 1e-7 or abs(x) > 9_999_999:
        return f"{x:.3e}"
    if x == 0:
        return "0"
    return f"{x:,.4f}"

stats = (
    df.groupby(['start_mode', 'algo', 'strategy'])['final_cash']
    .agg(count='count', mean='mean', std='std', var='var', min='min', median='median', max='max')
)
display(stats.map(fmt))

# %%
df = pd.concat([fixed_results, random_results], ignore_index=True)
# df = pd.concat([fixed_results], ignore_index=True)

# Flip fitness because gps is maximizing not minimizing
df.loc[df["algo"] == "gps", "best_fitness"] *= -1

def fmt(x):
    if pd.isna(x):
        return "NaN"
    if abs(x) < 1e-7 or abs(x) > 9_999_999:
        return f"{x:.3e}"
    if x == 0:
        return "0"
    return f"{x:,.4f}"

stats = (
    df.groupby(['algo', 'strategy'])['test_final_cash']
    .agg(count='count', mean='mean', std='std', var='var', min='min', median='median', max='max')
)

stats['cv'] = stats['std'] / stats['mean']

column_order = ['count', 'mean', 'std', 'cv', 'var', 'min', 'median', 'max']
stats = stats[column_order]

display(stats.map(fmt))


# %% [markdown]
# Sort by Strategy and CV for only fixed point initialization

# %%
df = pd.concat([fixed_results, random_results], ignore_index=True)

# Flip fitness because gps is maximizing not minimizing
df.loc[df["algo"] == "gps", "best_fitness"] *= -1

def fmt(x):
    if pd.isna(x):
        return "NaN"
    if abs(x) < 1e-7 or abs(x) > 9_999_999:
        return f"{x:.3e}"
    if x == 0:
        return "0"
    return f"{x:,.4f}"

stats = (
    df.groupby(['algo', 'strategy','start_mode'])['test_final_cash']
    .agg(count='count', mean='mean', std='std', var='var', min='min', median='median', max='max')
)

stats['cv'] = stats['std'] / stats['mean']


stats = stats.sort_values(by=[('strategy'), 'cv'], ascending=[True, True])

fixed_stats = stats.xs('fixed', level='start_mode')

column_order = ['count', 'mean', 'std', 'cv', 'min', 'median', 'max']


# Display formatted table
display(fixed_stats.map(fmt))

# %% [markdown]
# On Training Performance

# %%
df = pd.concat([fixed_results, random_results], ignore_index=True)
# df = pd.concat([fixed_results], ignore_index=True)


# Flip fitness because gps is maximizing not minimizing
df.loc[df["algo"] == "gps", "best_fitness"] *= -1

def fmt(x):
    if pd.isna(x):
        return "NaN"
    if abs(x) < 1e-7 or abs(x) > 9_999_999:
        return f"{x:.3e}"
    if x == 0:
        return "0"
    return f"{x:,.4f}"

stats = (
    df.groupby(['algo', 'strategy'])['final_cash']
    .agg(count='count', mean='mean', std='std', var='var', min='min', median='median', max='max')
)

stats['cv'] = stats['std'] / stats['mean']

column_order = ['count', 'mean', 'std', 'cv', 'var', 'min', 'median', 'max']
stats = stats[column_order]

display(stats.map(fmt))

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', context='notebook', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# %%
df = pd.concat([fixed_results, random_results], ignore_index=True)

# Flip fitness because gps is maximizing not minimizing
df.loc[df["algo"] == "gps", "best_fitness"] *= -1

# %% [markdown]
# # Kruskal Wallis

# %%
from scipy.stats import kruskal
import pandas as pd

results = []

for strategy in df['strategy'].unique():
    for init in df['start_mode'].unique():
        subset = df.query("strategy == @strategy and start_mode == @init")
        groups = [g['best_fitness'].values for _, g in subset.groupby('algo')]
        
        stat, p = kruskal(*groups)
        results.append({
            'strategy': strategy,
            'start_mode': init,
            'H_stat': stat,
            'p_value': p,
            'significant': p < 0.05
        })

results_df = pd.DataFrame(results)
display(results_df)

# %%
from scipy.stats import kruskal
import pandas as pd

results = []

for strategy in df['strategy'].unique():
    for init in df['start_mode'].unique():
        subset = df.query("strategy == @strategy and start_mode == @init")
        groups = [g['test_final_cash'].values for _, g in subset.groupby('algo')]
        
        stat, p = kruskal(*groups)
        results.append({
            'strategy': strategy,
            'start_mode': init,
            'H_stat': stat,
            'p_value': p,
            'significant': p < 0.05
        })

results_df = pd.DataFrame(results)
display(results_df)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scikit_posthocs import posthoc_dunn

def paired_posthoc(value_column):
    strategies = df['strategy'].unique()
    start_modes = df['start_mode'].unique()

    combinations = [(s, m) for s in strategies for m in start_modes]
    n_combos = len(combinations)

    # Grid layout
    ncols = 2
    nrows = (n_combos + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 5))
    axes = axes.flatten()

    for i, (strategy, start_mode) in enumerate(combinations):
        subset = df.query("strategy == @strategy and start_mode == @start_mode")
        
        dunn_result = posthoc_dunn(
            subset,
            val_col=value_column,
            group_col='algo',
            p_adjust='bonferroni'
        )
        
        mask = np.triu(np.ones_like(dunn_result, dtype=bool))
        
        sns.heatmap(
            dunn_result,
            mask=mask,
            annot=True,
            fmt='.2e',
            cmap='RdYlGn',
            vmin=0,
            vmax=0.05,
            linewidths=0.5,
            ax=axes[i],
            cbar_kws={'label': 'p-value'},
            annot_kws={'size': 8}
        )
        
        axes[i].set_title(
            f"strategy={strategy} | init={start_mode}",
            fontsize=10, fontweight='bold'
        )
        axes[i].tick_params(axis='x', rotation=45, labelsize=8)
        axes[i].tick_params(axis='y', rotation=0, labelsize=8)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Dunn's Post-hoc Pairwise Test — All Strategy × Init Combinations\n"
        "Red = significantly different (p < 0.05) | Green = no significant difference",
        fontsize=12, fontweight='bold', y=1.02
    )

    plt.tight_layout()
    if(value_column == 'final_cash'):
        plt.savefig('./report/images/dunn_pairwise_train.png', dpi=150, bbox_inches='tight')
    else:
        plt.savefig('./report/images/dunn_pairwise_test.png', dpi=150, bbox_inches='tight')
    plt.show()


# %%
paired_posthoc('final_cash')

# %%
paired_posthoc('test_final_cash')

# %%
# for strategy in df['strategy'].unique():
#     for init in df['start_mode'].unique():
#         subset = df.query("strategy == @strategy and start_mode == @init")
#         medians = subset.groupby('algo')['best_fitness'].median().sort_values(ascending=False)
#         print(f"\n{strategy} | {init}")
#         display(medians)

# %% [markdown]
# Calculate the ratio between result medians with buy and hold benchmark

# %%
df['bnh_ratio'] = df['test_final_cash'] / BUY_AND_HOLD_BENCHMARK

# %% [markdown]
# Check how many passes the benchmark

# %%
passing_benchmark_df = df[df['bnh_ratio'] > 1.0].sort_values(by='bnh_ratio')


# %%
def check_how_many_passes(df, group_by="algo"):
    fixed_df = df[df["start_mode"] == "fixed"]
    random_df = df[df["start_mode"] == "random"]

    fixed_counts = fixed_df.groupby(group_by).size()
    random_counts = random_df.groupby(group_by).size()
    fixed_var = fixed_df.groupby(group_by)["bnh_ratio"].var()
    random_var = random_df.groupby(group_by)["bnh_ratio"].var()

    summary_df = pd.DataFrame(
        {"fixed_count": fixed_counts, "random_count": random_counts,
         "fixed_var": fixed_var, "random_var": random_var}
    )

    # Force the specific algorithms and order (fills missing with NaN)
    target = ["abo", "aos", "gps", "pso", "sos", "mrfo"] if group_by == 'algo' \
        else ['sma', 'lma', 'ema_shared', 'ema_independent', 'weighted']

    summary_df = summary_df.reindex(target)

    # Fill NaN values with 0 for counts, convert counts to integers
    summary_df[["fixed_count", "random_count"]] = summary_df[["fixed_count", "random_count"]].fillna(0).astype(int)

    # Calculate percentages (divided by 1500)
    summary_df[f"% out of 1500 fixed"] = (summary_df["fixed_count"] / 1500) * 100
    summary_df[f"% out of 1500 random"] = (summary_df["random_count"] / 1500) * 100

    # Calculate percentage per trial (divided by 30)
    summary_df[f"% out of 50 fixed"] = (summary_df["fixed_count"] / 50) * 100
    summary_df[f"% out of 50 random"] = (summary_df["random_count"] / 50) * 100

    # Clean up column order for easy reading
    column_order = [
        "fixed_count",
        f"% out of 1500 fixed",
        f"% out of 50 fixed",
        "fixed_var",
        "random_count",
        f"% out of 1500 random",
        f"% out of 50 random",
        "random_var",
    ]

    summary_df = summary_df[column_order]
    return summary_df


display(check_how_many_passes(passing_benchmark_df, "algo").round(3))
display(check_how_many_passes(passing_benchmark_df, "strategy").round(3))

# %%
ct = pd.crosstab(
    passing_benchmark_df['algo'],
    [passing_benchmark_df['strategy'], passing_benchmark_df['start_mode']]
)

# Add total row
ct.loc['total'] = ct.sum()

display(ct)


# %% [markdown]
# # Plot the bar for test_final_cash

# %%

def plot_bar(data_points, plot_type, init_point, save_name):
    fig, ax = plt.subplots(figsize=(12, 5))

    sns.barplot(
        data=data_points,
        x='algo',
        y='bnh_ratio',
        hue='strategy',
        palette='colorblind',   # distinguishable without relying on red/green
        ax=ax
    )

    # Reference line
    ax.axhline(1, color='black', linewidth=0.8, linestyle='--', label='baseline buy and hold')

    ax.set_title(f'{plot_type} by algorithm and strategy ({init_point} init)', pad=12)
    ax.set_xlabel('Algorithm')
    ax.set_ylabel(f'{plot_type}')
    ax.legend(title='Strategy', bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.2f}x'))

    plt.tight_layout()
    plt.savefig(f'./report/images/{save_name}.png', dpi=150, bbox_inches='tight')
    plt.show()

def group_data(df, initial_mode='fixed', group_type='median'):
    grouped_data = (
        df[df['start_mode'] == initial_mode]
        .groupby(['algo', 'strategy'])['bnh_ratio']
        .agg(group_type)
        .reset_index()
    )
    return grouped_data

plot_bar(group_data(df, initial_mode='fixed', group_type='median'),
    plot_type='Median Test Cash', init_point='fixed', save_name='max_test_cash_random')

plot_bar(group_data(df, initial_mode='fixed', group_type='max'),
    plot_type='Max Test Cash', init_point='fixed', save_name='max_test_cash_fixed')

plot_bar(group_data(df, initial_mode='random', group_type='median'),
    plot_type='Median Test Cash', init_point='random', save_name='median_test_cash_random')
    
plot_bar(group_data(df, initial_mode='random', group_type='max'),
    plot_type='Max Test Cash', init_point='random', save_name='median_test_cash_fixed')

# %% [markdown]
# ## Spearman ρ(train_fitness, test_fitness) per Strategy

# %% [markdown]
# This is pooled across both fixed and random

# %%
from scipy.stats import spearmanr

strategies = ["sma", "lma", "ema_shared", "ema_independent", "weighted"]

rhos, pvals = [], []
for s in strategies:
    sub = df[df["strategy"] == s]
    rho, p = spearmanr(sub["final_cash"], sub["test_final_cash"])
    rhos.append(rho)
    pvals.append(p)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(strategies, rhos, color="steelblue", edgecolor="white", linewidth=0.8, zorder=3)

for bar, rho, p in zip(bars, rhos, pvals):
    yoff = 0.03 if rho >= 0 else -0.06

    p_text = f"{p:.1e}" if p < 0.001 else f"{p:.3f}"

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        rho + yoff,
        f'rho:{round(rho, 2)}\nsig: {p_text}',
        ha="center", va="bottom", fontsize=8,
    )

ax.axhline(0, color="black", linestyle="--", linewidth=1.2, label="rho = 0 (no transfer)", zorder=2)

ax.set_ylabel("Spearman rho  (train  vs  test  fitness)", fontsize=11)
ax.set_xlabel("Strategy", fontsize=11)
ax.set_title("Train vs Test Fitness Transfer by Strategy", fontsize=13)
ax.set_ylim(min(rhos) - 0.18, max(rhos) + 0.18)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.set_xticklabels(strategies, rotation=15, ha="right")

# Significance legend
from matplotlib.lines import Line2D
legend_items = [
    Line2D([0], [0], color="none", label="*** p < 0.001"),
    Line2D([0], [0], color="none", label="**  p < 0.01"),
    Line2D([0], [0], color="none", label="*   p < 0.05"),
    Line2D([0], [0], color="none", label="ns  p ≥ 0.05"),
]
ax.legend(handles=[ax.get_legend_handles_labels()[0][0]] + legend_items,
          labels=["rho = 0 (no transfer)"],
          fontsize=9, loc="upper left")

plt.tight_layout()
plt.savefig("./report/images/train_test_spearman.png", dpi=150)
plt.show()


# %% [markdown]
# When checking with spreadsheet, on `ema_shared` with `mrfo` on both random and fixed init there are 2 entries with 5660ish profit that got excluded from the result. Those are mrfo run 14 (fixed) and 1 (random).
#
# Let's try visualizing the plot
#
#

# %%
mrfo_14 = df.query('algo == "mrfo" & strategy == "ema_shared" & run_id == 14 & start_mode == "fixed"')
mrfo_1 = df.query('algo == "mrfo" & strategy == "ema_shared" & run_id == 14 & start_mode == "random"')

from runners.mrfo.plot import plot_equity_and_purchases
from runners.mrfo.common import re_evaluate
from runners.mrfo.ema import get_signals_shared
import json

best_params_json = mrfo_1["best_params"].iloc[0]
short, long, buy_at, sell_at, equity_curve, cash = \
    re_evaluate(json.loads(best_params_json), train_prices, get_signals_shared)

sns.lineplot(mrfo_1["equity_curve"].iloc[0])

# %%
pso_15 = df.query('algo == "pso" & strategy == "ema_shared" & run_id == 15 & start_mode == "fixed"')

from runners.pso.common import re_evaluate
from runners.pso.ema import get_signals_shared
import json

best_params_json = pso_15["best_params"].iloc[0]
short, long, buy_at, sell_at, equity_curve, cash = \
    re_evaluate(json.loads(best_params_json), train_prices, get_signals_shared)

print(buy_at)
print(sell_at)

# %% [markdown]
# # Aggregate and Save

# %%
# agg_funcs = ["mean", "median", "std", lambda x: x.std() / x.mean(), "min", "max"]
# agg_names  = ["mean", "median", "std", "cv", "min", "max"]

# def summarise(df, cols):
#     return (
#         df.groupby(["strategy", "algo"])[cols]
#         .agg(agg_funcs)
#         .set_axis(
#             pd.MultiIndex.from_product([cols, agg_names]),
#             axis=1,
#         )
#     )

# # Table 1 — cash performance
# print("=== Fixed — Cash ===")
# display(summarise(fixed_results, ["final_cash", "test_final_cash"]))

# print("=== Random — Cash ===")
# display(summarise(random_results, ["final_cash", "test_final_cash"]))

# # Table 2 — runtime & epochs
# print("=== Fixed — Runtime / Epochs ===")
# display(summarise(fixed_results, ["runtime_ms", "epoch_count"]))

# print("=== Random — Runtime / Epochs ===")
# display(summarise(random_results, ["runtime_ms", "epoch_count"]))

# %%
# summary_fixed_cash    = summarise(fixed_results,  ["final_cash", "test_final_cash"])
# summary_random_cash   = summarise(random_results, ["final_cash", "test_final_cash"])
# summary_fixed_runtime = summarise(fixed_results,  ["runtime_ms", "epoch_count"])
# summary_random_runtime = summarise(random_results, ["runtime_ms", "epoch_count"])

# summary_fixed_cash.to_csv("results/summary_fixed_cash.csv")
# summary_random_cash.to_csv("results/summary_random_cash.csv")
# summary_fixed_runtime.to_csv("results/summary_fixed_runtime.csv")
# summary_random_runtime.to_csv("results/summary_random_runtime.csv")

# print("Summaries saved to results/")

# %% [markdown]
# ## Reproduce a single run (retrain + plot)

# %%
from experiment_runner import reproduce_result, plot_reproduce_result

for run_id in (8, 40, 7, 48):
    mode = "fixed" if run_id in (8, 40) else "random"
    result = reproduce_result("gps", "ema_shared", run_id=run_id, start_mode=mode)
    plot_reproduce_result(result, split="both")



# %%
from experiment_runner import reproduce_result, plot_reproduce_result

result = reproduce_result("gps", "ema_independent", run_id=27, start_mode="fixed")
plot_reproduce_result(result, split="both")



# %%
from experiment_runner import reproduce_result, plot_reproduce_result

result = reproduce_result("abo", "ema_shared", run_id=20, start_mode="random")
plot_reproduce_result(result, split="both")



# %%
from experiment_runner import reproduce_result, plot_reproduce_result

result = reproduce_result("sos", "ema_shared", run_id=18, start_mode="fixed")
plot_reproduce_result(result, split="both")



# %%
from experiment_runner import reproduce_result, plot_reproduce_result

result = reproduce_result("mrfo", "weighted", run_id=0, start_mode="random")
plot_reproduce_result(result, split="both")


