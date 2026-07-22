import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

INPUT_FILE = 'loops/loop_df_3kb.tsv'
OUTPUT_DIR = 'models/transform_validation'
ABLE_COL   = 'AbLE_score'


def load_and_clean(path: str, able_col: str = ABLE_COL) -> pd.DataFrame:
    print(f"Reading {path} ...")
    df = pd.read_csv(path, sep='\t')
    n_raw = len(df)

    n_nan    = df[able_col].isna().sum()
    n_nonpos = (df[able_col] <= 0).sum()
    df = df.dropna(subset=[able_col])
    df = df[df[able_col] > 0].copy()

    print(f"Raw rows: {n_raw:,}")
    print(f"Dropped NaN: {n_nan:,}")
    print(f"Dropped <= 0: {n_nonpos:,}")
    print(f"Remaining: {len(df):,}  (all {able_col} > 0)")

    if (~np.isfinite(df[able_col])).any():
        raise ValueError(f"Non-finite {able_col} values remain after filtering.")
    return df


def plot_log_able_diagnostic(log_able: pd.Series, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.hist(log_able, bins=100, color='coral', edgecolor='none', alpha=0.85)
    ax.set_xlabel('log(AbLE_score)')
    ax.set_ylabel('Count')
    ax.set_title(f'log(AbLE_score)  [n={len(log_able):,}]\n'
                 f'skewness={log_able.skew():.2f}  '
                 f'median={log_able.median():.2f}')

    ax = axes[1]
    samp = log_able.sample(min(5000, len(log_able)), random_state=42)
    (osm, osr), (slope, intercept, _) = stats.probplot(samp, dist='norm')
    ax.plot(osm, osr, '.', ms=2, alpha=0.35, color='coral', rasterized=True)
    ax.plot(osm, slope * np.array(osm) + intercept, 'r-', lw=1.5)
    W, p = stats.shapiro(samp)
    ax.set_title(f'Q-Q: log(AbLE_score)\nShapiro W={W:.3f}  p={p:.1e}')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel('Sample quantiles')

    path = os.path.join(save_dir, 'log_able_diagnostic.svg')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved -> {path}")


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df       = load_and_clean(INPUT_FILE)
    log_able = np.log(df[ABLE_COL])

    plot_log_able_diagnostic(log_able, save_dir=OUTPUT_DIR)
