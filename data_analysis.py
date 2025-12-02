import os
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

INPUT_CSV = "raw_results.csv"
OUT_DIR = "analysis_figures"
ZIP_NAME = "analysis_figures.zip"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def loglog_regression(x, y):
    """
    Perform linear regression on (log x, log y).
    Returns dict: slope, intercept, r2, mask (boolean array of valid points)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    mask = (x > 0) & (y > 0)
    if mask.sum() < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "mask": mask}
    lx = np.log(x[mask]).reshape(-1, 1)
    ly = np.log(y[mask])
    reg = LinearRegression().fit(lx, ly)
    return {
        "slope": float(reg.coef_[0]),
        "intercept": float(reg.intercept_),
        "r2": float(reg.score(lx, ly)),
        "mask": mask
    }


def save_fig(fig, out_path, dpi=150):
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def style_axes(ax):
    # remove interior grid lines and make axes (spines/ticks) pure black
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", which="both",
                   colors="black", labelcolor="black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    ax.title.set_color("black")
    ax.set_facecolor("white")


def plot_loglog_with_fit(ax, x, y, res, color, marker='x', label=None):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = res["mask"]
    ax.scatter(x[mask], y[mask], marker=marker, s=40, edgecolor='k', linewidth=0.3,
               alpha=0.85, color=color, label=label, zorder=3)

    if mask.sum() > 1 and not math.isnan(res["slope"]):
        lx = np.log(x[mask])
        ly = np.log(y[mask])
        ly_hat = res["intercept"] + res["slope"] * lx
        resid = ly - ly_hat
        std = np.std(resid, ddof=1) if len(resid) > 1 else 0.0

        order = np.argsort(x[mask])
        xs = x[mask][order]
        lyh_sorted = ly_hat[order]
        ys_line = np.exp(lyh_sorted)
        lower = np.exp(lyh_sorted - std)
        upper = np.exp(lyh_sorted + std)

        ax.plot(xs, ys_line, linestyle="--", linewidth=2.0, alpha=0.95,
                color=color, zorder=4, label=f"fit slope={res['slope']:.3f} R2={res['r2']:.3f}")
        ax.fill_between(xs, lower, upper, alpha=0.22,
                        color=color, linewidth=0, zorder=2)


def main():
    ensure_dir(OUT_DIR)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found at '{INPUT_CSV}'.")
    df = pd.read_csv(INPUT_CSV)

    sns.set_style("white")
    sns.set_context("talk", rc={"axes.titlesize": 16,
                    "axes.labelsize": 13, "legend.fontsize": 11})

    iso_color = "#455A70"
    ks_color = "#7E546F"
    topo_palette = ["#FFC759", "#FF7B9C", "#607196", "#BABFD1", "#E8E9ED"]

    graph_types = df["Graph_Type"].unique()
    topo_colors = {gt: topo_palette[i % len(
        topo_palette)] for i, gt in enumerate(graph_types)}

    summary_rows = []
    loglog_rows = []

    # For master/comparative charts we will collect all data by topology
    master_data = []

    for idx, gt in enumerate(graph_types):
        sub = df[df["Graph_Type"] == gt].sort_values(
            "Nodes").reset_index(drop=True)

        res_iso = loglog_regression(
            sub["Pred_Iso"].values, sub["Time_Iso_Mean"].values)
        res_ks = loglog_regression(
            sub["Pred_KS"].values, sub["Time_KS_Mean"].values)

        summary_rows.append({
            "Graph_Type": gt,
            "Iso_slope": res_iso["slope"],
            "Iso_r2": res_iso["r2"],
            "KS_slope": res_ks["slope"],
            "KS_r2": res_ks["r2"],
        })

        loglog_rows.append({
            "Graph_Type": gt,
            "Algorithm": "Iso",
            "Slope(log-log)": res_iso["slope"],
            "Intercept(log-log)": res_iso["intercept"],
            "R2": res_iso["r2"]
        })
        loglog_rows.append({
            "Graph_Type": gt,
            "Algorithm": "KS",
            "Slope(log-log)": res_ks["slope"],
            "Intercept(log-log)": res_ks["intercept"],
            "R2": res_ks["r2"]
        })

        topo_color = topo_colors[gt]

        # Save for master chart
        master_data.append({
            "Graph_Type": gt,
            "Pred_Iso": sub["Pred_Iso"].values,
            "Time_Iso": sub["Time_Iso_Mean"].values,
            "Pred_KS": sub["Pred_KS"].values,
            "Time_KS": sub["Time_KS_Mean"].values,
            "Nodes": sub["Nodes"].values,
            "color": topo_color
        })

        # log-log scatter plots with linear fit (Iso and KS)
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        # Iso
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].set_xlabel("Pred_Iso (log scale)")
        axs[0].set_ylabel("Time_Iso_Mean (log scale)")
        axs[0].set_title(f"Iso: Time vs Predictor ({gt})")
        style_axes(axs[0])
        plot_loglog_with_fit(axs[0], sub["Pred_Iso"].values, sub["Time_Iso_Mean"].values,
                             res_iso, iso_color, marker="o", label="Iso data")
        axs[0].legend(frameon=True, facecolor="white", edgecolor="0.8")

        # KS
        axs[1].set_xscale("log")
        axs[1].set_yscale("log")
        axs[1].set_xlabel("Pred_KS (log scale)")
        axs[1].set_ylabel("Time_KS_Mean (log scale)")
        axs[1].set_title(f"KS: Time vs Predictor ({gt})")
        style_axes(axs[1])
        plot_loglog_with_fit(axs[1], sub["Pred_KS"].values, sub["Time_KS_Mean"].values,
                             res_ks, ks_color, marker="o", label="KS data")
        axs[1].legend(frameon=True, facecolor="white", edgecolor="0.8")

        fig.suptitle(f"{gt} — topology accent", fontsize=18, y=0.99)
        fig.patch.set_facecolor("white")
        axs[0].add_patch(plt.Rectangle((0.02, 0.95), 0.18, 0.03, transform=fig.transFigure,
                                       facecolor=topo_color, edgecolor="none", zorder=1))

        save_fig(fig, os.path.join(
            OUT_DIR, f"{gt}_predictor_scatter_loglog.png"))

        # simple time vs nodes plot, both algorithms
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.plot(sub["Nodes"], sub["Time_Iso_Mean"], marker="o", label="Iso",
                color=iso_color, linewidth=2, alpha=0.95)
        ax.plot(sub["Nodes"], sub["Time_KS_Mean"], marker="s", label="KS",
                color=ks_color, linewidth=2, alpha=0.95)
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Time vs Nodes ({gt})")
        style_axes(ax)
        ax.legend(frameon=True)
        save_fig(fig, os.path.join(OUT_DIR, f"{gt}_time_vs_nodes.png"))

        # normalized time = time / predictor plot
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        iso_norm = sub["Time_Iso_Mean"] / sub["Pred_Iso"].replace(0, np.nan)
        ks_norm = sub["Time_KS_Mean"] / sub["Pred_KS"].replace(0, np.nan)
        ax.plot(sub["Nodes"], iso_norm, marker="o",
                label="Iso time / Pred_Iso", color=iso_color, linewidth=2, alpha=0.95)
        ax.plot(sub["Nodes"], ks_norm, marker="s", label="KS time / Pred_KS",
                color=ks_color, linewidth=2, alpha=0.95)
        ax.set_xlabel("Nodes")
        ax.set_ylabel("Time / Predictor")
        ax.set_title(f"Normalized time (time/predictor) vs Nodes ({gt})")
        style_axes(ax)
        ax.legend(frameon=True)
        save_fig(fig, os.path.join(OUT_DIR, f"{gt}_time_over_predictor.png"))

        # theoretical fit of KS using n^2 * (log2 n)^3
        n = sub["Nodes"].values.astype(float)
        n_pos = n.copy()
        n_pos[n_pos < 2] = 2.0
        ks_theory = (n_pos ** 2) * (np.log2(n_pos) ** 3)

        ks_time = sub["Time_KS_Mean"].values

        mask_nonneg = ~np.isnan(ks_time)
        X = ks_theory[mask_nonneg].reshape(-1, 1)
        Y = ks_time[mask_nonneg].reshape(-1, 1)
        if len(X) >= 2:
            reg_lin = LinearRegression().fit(X, Y)
            coef = float(reg_lin.coef_[0][0])
            intercept = float(reg_lin.intercept_[0])
            r2_lin = float(reg_lin.score(X, Y))
        else:
            coef, intercept, r2_lin = np.nan, np.nan, np.nan

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.scatter(ks_theory, ks_time, marker="x", s=50,
                   color=ks_color, edgecolor='k', alpha=0.9, label="KS time")
        if not math.isnan(coef):
            xs = np.logspace(math.log10(max(ks_theory.min(), 1e-6)),
                             math.log10(max(ks_theory.max(), 1.0)), 200)
            ax.plot(xs, coef * xs + intercept, linestyle="--", linewidth=2.0,
                    label=f"linear fit: T ≈ {coef:.3e}*theory + {intercept:.3f}, R2={r2_lin:.3f}",
                    color=ks_color, alpha=0.95)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n^2 * (log2 n)^3 (log scale)")
        ax.set_ylabel("Time_KS_Mean (log scale)")
        ax.set_title(f"KS time vs n^2 (log^3) fit ({gt})")
        style_axes(ax)
        ax.legend(frameon=True)
        save_fig(fig, os.path.join(OUT_DIR, f"{gt}_ks_theory_fit.png"))

        # Additional per-topology analytic charts:
        # 1) Algorithms vs KS theoretical predictor
        fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(ks_theory, sub["Time_Iso_Mean"], marker="o", s=50,
                   color=iso_color, edgecolor='k', alpha=0.9, label="Iso time")
        ax.scatter(ks_theory, sub["Time_KS_Mean"], marker="s", s=50,
                   color=ks_color, edgecolor='k', alpha=0.9, label="KS time")
        ax.set_xlabel("n^2 * (log2 n)^3 (log scale)")
        ax.set_ylabel("Time (log scale)")
        ax.set_title(f"Iso & KS times vs KS-theory ({gt})")
        style_axes(ax)
        ax.legend(frameon=True)
        save_fig(fig, os.path.join(OUT_DIR, f"{gt}_alg_vs_ks_theory.png"))

        # 2) Direct comparison: Time_Iso vs Time_KS
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.scatter(sub["Time_Iso_Mean"], sub["Time_KS_Mean"], marker="o", s=60,
                   color=topo_color, edgecolor='k', alpha=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time_Iso_Mean (log)")
        ax.set_ylabel("Time_KS_Mean (log)")
        ax.set_title(f"Iso time vs KS time ({gt})")
        style_axes(ax)
        # identity line
        minv = min(sub["Time_Iso_Mean"].min(), sub["Time_KS_Mean"].min())
        maxv = max(sub["Time_Iso_Mean"].max(), sub["Time_KS_Mean"].max())
        if np.isfinite(minv) and np.isfinite(maxv) and minv > 0 and maxv > 0:
            xs = np.logspace(math.log10(minv), math.log10(maxv), 100)
            ax.plot(xs, xs, linestyle="--", color="gray",
                    linewidth=1.0, zorder=2)
        save_fig(fig, os.path.join(OUT_DIR, f"{gt}_iso_vs_ks_scatter.png"))

    # Master chart: overlay Pred vs Time log-log for all topologies and both algorithms
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for item in master_data:
        gt = item["Graph_Type"]
        color = item["color"]
        ax.scatter(item["Pred_Iso"], item["Time_Iso"], marker="o", s=50,
                   edgecolor='k', linewidth=0.3, alpha=0.9, color=color, label=f"{gt} Iso")
        ax.scatter(item["Pred_KS"], item["Time_KS"], marker="s", s=50,
                   edgecolor='k', linewidth=0.3, alpha=0.9, color=color, label=f"{gt} KS", facecolors='none')
    ax.set_xlabel("Predictor (log scale)")
    ax.set_ylabel("Time (log scale)")
    ax.set_title("Master chart: Predictor vs Time (all topologies, log-log)")
    style_axes(ax)
    # create a legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=True, ncol=2)
    save_fig(fig, os.path.join(
        OUT_DIR, "master_predictor_scatter_loglog_all_topologies.png"))

    # Same-algorithm across topologies (Iso)
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for item in master_data:
        gt = item["Graph_Type"]
        color = item["color"]
        ax.scatter(item["Pred_Iso"], item["Time_Iso"], marker="o", s=50,
                   edgecolor='k', linewidth=0.3, alpha=0.9, color=color, label=gt)
    ax.set_xlabel("Pred_Iso (log scale)")
    ax.set_ylabel("Time_Iso_Mean (log scale)")
    ax.set_title("Iso across topologies (log-log)")
    style_axes(ax)
    ax.legend(frameon=True, ncol=2)
    save_fig(fig, os.path.join(OUT_DIR, "iso_across_topologies.png"))

    # Comprehensive: Iso on first topology vs KS on second topology
    if len(master_data) >= 2:
        a = master_data[0]
        b = master_data[1]
        fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.scatter(a["Pred_Iso"], a["Time_Iso"], marker="o", s=60,
                   edgecolor='k', linewidth=0.3, alpha=0.9, color=a["color"], label=f"{a['Graph_Type']} Iso")
        ax.scatter(b["Pred_KS"], b["Time_KS"], marker="s", s=60,
                   edgecolor='k', linewidth=0.3, alpha=0.9, color=b["color"], label=f"{b['Graph_Type']} KS")
        ax.set_xlabel("Predictor (log scale)")
        ax.set_ylabel("Time (log scale)")
        ax.set_title(
            f"Comprehensive: {a['Graph_Type']} Iso vs {b['Graph_Type']} KS (log-log)")
        style_axes(ax)
        ax.legend(frameon=True)
        save_fig(fig, os.path.join(
            OUT_DIR, "comprehensive_iso_first_vs_ks_second.png"))

    # Save CSV summaries
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "loglog_summary.csv"), index=False)

    slopes_df = pd.DataFrame(loglog_rows)
    slopes_df.to_csv(os.path.join(
        OUT_DIR, "loglog_slopes_full.csv"), index=False)

    # Create a zip of all outputs
    zip_path = os.path.join(OUT_DIR, ZIP_NAME)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(OUT_DIR):
            for f in files:
                if f == ZIP_NAME:
                    continue
                z.write(os.path.join(root, f), arcname=f)

    print("DONE")
    print(f"Outputs written to folder: {OUT_DIR}")
    print(f"Zip archive: {zip_path}")
    print("Files included:")
    for root, _, files in os.walk(OUT_DIR):
        for f in sorted(files):
            print(" -", f)


if __name__ == "__main__":
    main()
