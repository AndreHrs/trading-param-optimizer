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
#     display_name: cits5508
#     language: python
#     name: python3
# ---

# %%
from utilities.data_loader import load_data

train_prices, train_dates, test_prices, test_dates = load_data("./data/BTC-Daily.csv")

# %%
import pandas as pd

fixed_results = pd.read_csv("results/fixed_runs.csv")
random_results = pd.read_csv("results/random_runs.csv")

# %%
df = pd.concat([fixed_results, random_results], ignore_index=True)
display(df.groupby(['start_mode', 'algo', 'strategy'])['test_final_cash'].describe())
display(df.dtypes)
display(df.isnull().sum())
df['algo'].value_counts()
df['strategy'].value_counts()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Set once at the top of your notebook — applies to everything below
sns.set_theme(style='whitegrid', context='notebook', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 120,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# %% [markdown]
# # Plot the bar for test_final_cash

# %%

# Then your plot
fig, ax = plt.subplots(figsize=(12, 5))

medians = (df[df['start_mode'] == 'fixed']
           .groupby(['algo', 'strategy'])['test_final_cash']
           .median()
           .reset_index())

sns.barplot(
    data=medians,
    x='algo',
    y='test_final_cash',
    hue='strategy',
    palette='colorblind',   # distinguishable without relying on red/green
    ax=ax
)

# Reference line — starting capital
ax.axhline(1000, color='black', linewidth=0.8, linestyle='--', label='starting capital')

ax.set_title('Median test cash by algorithm and strategy (fixed init)', pad=12)
ax.set_xlabel('Algorithm')
ax.set_ylabel('Median test cash (USD)')
ax.legend(title='Strategy', bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

plt.tight_layout()
plt.show()

# %%
# Pivot so fixed and random are columns, then compute difference
diff = (df.groupby(['start_mode', 'algo', 'strategy'])['test_final_cash']
        .median()
        .unstack('start_mode')   # start_mode becomes columns
        .reset_index())

diff['diff'] = diff['random'] - diff['fixed']

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=diff, x='algo', y='diff', hue='strategy', palette='colorblind', ax=ax)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.set_title('Random vs fixed init: difference in median test cash')
ax.set_ylabel('Random - Fixed (USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend(title='Strategy', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()

# %% [markdown]
# # Plot boxplot

# %%
strategies_of_interest = ['sma', 'lma', 'ema_independent', 'weighted']

g = sns.FacetGrid(
    df[(df['start_mode'] == 'fixed') & 
       (df['strategy'].isin(strategies_of_interest))],
    col='strategy', col_wrap=2, height=4, aspect=1.3,
    sharey=False
)

g.map_dataframe(
    sns.boxplot,
    x='algo',
    y='test_final_cash',
    hue='algo',           # color by algo
    palette='colorblind',
    legend=True          # suppress per-subplot legends, we'll add one
)

g.set_xticklabels(rotation=45, ha='right')   # ha='right' stops labels from overlapping
g.set_titles(col_template='strategy = {col_name}')
g.set_axis_labels('Algorithm', 'Test cash (USD)')

# Add a shared legend
g.add_legend(title='Algorithm')

plt.tight_layout()

# %% [markdown]
# # Train vs Test Scatter

# %%
sns.scatterplot(
    data=df[df['start_mode'] == 'fixed'],
    x='final_cash',
    y='test_final_cash',
    hue='algo',
    style='strategy',
    alpha=0.6
)

# Add the diagonal — points on this line have zero overfit
import numpy as np
lim = df['final_cash'].max()
plt.plot([0, lim], [0, lim], 'k--', linewidth=0.8, label='train = test')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('train cash')
plt.ylabel('test cash')

# %% [markdown]
# # Rank Stability

# %%
# Compute rank within each strategy (per start_mode if you want to compare them)
fixed_df_ranked = df[df['start_mode'] == 'fixed'].copy()

fixed_df_ranked['rank'] = fixed_df_ranked.groupby('strategy')['test_final_cash'] \
                            .rank(ascending=False, method='average')

# Now plot — each dot is one run, spread by strategy
sns.stripplot(
    data=fixed_df_ranked,
    x='algo',
    y='rank',
    hue='strategy',
    dodge=True,       # separates hue groups horizontally
    alpha=0.5,
    jitter=True       # spreads overlapping points
)
plt.gca().invert_yaxis()  # rank 1 at top

# %% [markdown]
# # Runtime vs Performance

# %%
# Aggregate first — you want one point per algo/strategy, not 30 overlapping points
agg = df[df['start_mode'] == 'fixed'].groupby(['algo', 'strategy']).agg(
    test_median=('test_final_cash', 'median'),
    runtime_median=('runtime_ms', 'median')
).reset_index()

sns.scatterplot(
    data=agg,
    x='runtime_median',
    y='test_median',
    hue='algo',
    style='strategy',
    s=100
)

# Annotate outliers so you can read the plot
for _, row in agg.iterrows():
    if row['runtime_median'] > 15000 or row['test_median'] > 4000:
        plt.annotate(
            f"{row['algo']}\n{row['strategy']}",
            (row['runtime_median'], row['test_median']),
            fontsize=7, alpha=0.7
        )

# %% [markdown]
# # Cash per Time

# %%
agg['cash_per_ms'] = (agg['test_median'] - 1000) / agg['runtime_median']

sns.barplot(
    data=agg,
    x='algo',
    y='cash_per_ms',
    hue='strategy'
)
plt.ylabel('test profit per ms of runtime')

# %% [markdown]
# # Aggregate and Save

# %%
agg_funcs = ["mean", "median", "std", lambda x: x.std() / x.mean(), "min", "max"]
agg_names  = ["mean", "median", "std", "cv", "min", "max"]

def summarise(df, cols):
    return (
        df.groupby(["strategy", "algo"])[cols]
        .agg(agg_funcs)
        .set_axis(
            pd.MultiIndex.from_product([cols, agg_names]),
            axis=1,
        )
    )

# Table 1 — cash performance
print("=== Fixed — Cash ===")
display(summarise(fixed_results, ["final_cash", "test_final_cash"]))

print("=== Random — Cash ===")
display(summarise(random_results, ["final_cash", "test_final_cash"]))

# Table 2 — runtime & epochs
print("=== Fixed — Runtime / Epochs ===")
display(summarise(fixed_results, ["runtime_ms", "epoch_count"]))

print("=== Random — Runtime / Epochs ===")
display(summarise(random_results, ["runtime_ms", "epoch_count"]))

# %%
summary_fixed_cash    = summarise(fixed_results,  ["final_cash", "test_final_cash"])
summary_random_cash   = summarise(random_results, ["final_cash", "test_final_cash"])
summary_fixed_runtime = summarise(fixed_results,  ["runtime_ms", "epoch_count"])
summary_random_runtime = summarise(random_results, ["runtime_ms", "epoch_count"])

summary_fixed_cash.to_csv("results/summary_fixed_cash.csv")
summary_random_cash.to_csv("results/summary_random_cash.csv")
summary_fixed_runtime.to_csv("results/summary_fixed_runtime.csv")
summary_random_runtime.to_csv("results/summary_random_runtime.csv")

print("Summaries saved to results/")
