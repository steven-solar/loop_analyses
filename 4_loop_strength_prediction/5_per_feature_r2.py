# conda activate umap_env

"""
Linear regression (sklearn) of each loop feature against AbLE_score,
computed overall and per cell line, with scatter + fit-line plots.

Usage:
    python loop_feature_regression_by_cellline.py path/to/loops.csv
	

Expects a table (CSV/TSV) with at least these columns:
    H2Aub_left, H2Aub_right, Rad21_left, Rad21_right, AbLE_score
    + a cell-line identifier column (default name: "cell_line";
      override by passing it as the 2nd CLI arg if yours is named differently)

Outputs:
    regression_results.csv        - R2/slope/intercept per feature, overall + per cell line
    regression_plots_overall.png  - scatter + fit line per feature, all data pooled
    regression_plots_<cellline>.png - same, one file per cell line
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

FEATURES = ["H2Aub_left", "H2Aub_right", "Rad21_left", "Rad21_right", "CTCF_left", "CTCF_right"]
TARGET = "AbLE_score"


def load_table(path):
    sep = "\t" if path.lower().endswith((".tsv", ".txt")) else None
    df = pd.read_csv(path, sep=sep, engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df


def regress(df, feature, target):
    sub = df[[feature, target]].apply(pd.to_numeric, errors="coerce").dropna()
    n_dropped = len(df) - len(sub)
    if len(sub) < 3:
        return {"feature": feature, "n": len(sub), "n_dropped": n_dropped,
                "r2": np.nan, "slope": np.nan, "intercept": np.nan,
                "warning": "too few valid points", "x": None, "y": None, "model": None}

    X = sub[[feature]].values
    y = np.log(sub[target].values)
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)

    return {
        "feature": feature,
        "n": len(sub),
        "n_dropped": n_dropped,
        "r2": r2_score(y, pred),
        "slope": model.coef_[0],
        "intercept": model.intercept_,
        "warning": "",
        "x": X.ravel(),
        "y": y,
        "model": model,
    }


def plot_group(results, title, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, r in zip(axes.ravel(), results):
        if r["x"] is None:
            ax.set_title(f"{r['feature']} (skipped: {r['warning']})")
            continue
        x, y = r["x"], r["y"]
        ax.scatter(x, y, alpha=0.4, s=15, color="steelblue")
        xs = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        ax.plot(xs, r["model"].predict(xs), color="crimson", linewidth=2)
        ax.set_xlabel(r["feature"])
        ax.set_ylabel(TARGET)
        ax.set_title(f"{r['feature']} vs {TARGET}\nR2={r['r2']:.3f}, n={r['n']}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"[info] saved plot to {out_path}", file=sys.stderr)


def run_group(df, group_label):
    results = [regress(df, feat, TARGET) for feat in FEATURES]
    table_cols = ["feature", "n", "n_dropped", "r2", "slope", "intercept", "warning"]
    rows = [{"group": group_label, **{k: r[k] for k in table_cols}} for r in results]
    return results, rows


def main(path):
    df = load_table(path)

    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}\nFound columns: {list(df.columns)}")

    all_rows = []

    overall_results, overall_rows = run_group(df, "ALL")
    all_rows.extend(overall_rows)
    plot_group(overall_results, "All cell lines pooled", "regression_plots_overall.png")

    for cell_line, sub_df in df.groupby("cell_line"):
        results, rows = run_group(sub_df, str(cell_line))
        all_rows.extend(rows)
        safe_name = str(cell_line).replace("/", "_").replace(" ", "_")
        plot_group(results, f"Cell line: {cell_line}", f"regression_plots_{safe_name}.png")

    out = pd.DataFrame(all_rows)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")
    print(out.to_string(index=False))

    for _, r in out.iterrows():
        if r["n_dropped"] > 0:
            print(f"[warn] {r['group']}/{r['feature']}: dropped {r['n_dropped']} rows with non-numeric/NaN values", file=sys.stderr)
        if r["warning"]:
            print(f"[warn] {r['group']}/{r['feature']}: {r['warning']}", file=sys.stderr)

    out.to_csv("per_cl_reclustered/regression_results.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("Usage: python loop_feature_regression_by_cellline.py <table.csv>")
    main(sys.argv[1])