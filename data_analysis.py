import os
import math
import zipfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

INPUT_CSV = "raw_results.csv"
OUT_DIR = "analysis_figures"
ZIP_NAME = "analysis_figures.zip"


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    ensure_dir(path)


def loglog_regression(x, y):
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
    base, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".pdf"]:
        out_path = base + ".png"
        ext = ".png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    try:
        fig.savefig(base + ".pdf", dpi=dpi, bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def style_axes(ax):
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
    clear_dir(OUT_DIR)

    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input CSV not found at '{INPUT_CSV}'.")
    df = pd.read_csv(INPUT_CSV)

    iso_color = "#455A70"
    ks_color = "#7E546F"
    topo_palette = ["#FFC759", "#D94B6A", "#607196", "#BABFD1", "#87A878"]

    graph_types = df["Graph_Type"].unique()
    topo_colors = {gt: topo_palette[i % len(
        topo_palette)] for i, gt in enumerate(graph_types)}

    summary_rows = []
    loglog_rows = []
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

        m_actual = sub["Edges"].values.astype(float)
        iso_theory = m_actual * (np.log2(n_pos) ** 6)

        iso_time = sub["Time_Iso_Mean"].values
        mask_nonneg_iso = ~np.isnan(iso_time)
        X_iso = iso_theory[mask_nonneg_iso].reshape(-1, 1)
        Y_iso = iso_time[mask_nonneg_iso].reshape(-1, 1)
        if len(X_iso) >= 2:
            reg_iso = LinearRegression().fit(X_iso, Y_iso)
            coef_iso = float(reg_iso.coef_[0][0])
            intercept_iso = float(reg_iso.intercept_[0])
            r2_iso = float(reg_iso.score(X_iso, Y_iso))
        else:
            coef_iso, intercept_iso, r2_iso = np.nan, np.nan, np.nan

        master_data.append({
            "Graph_Type": gt,
            "Pred_Iso": sub["Pred_Iso"].values,
            "Time_Iso": sub["Time_Iso_Mean"].values,
            "Pred_KS": sub["Pred_KS"].values,
            "Time_KS": sub["Time_KS_Mean"].values,
            "Nodes": sub["Nodes"].values,
            "color": topo_color,
            "res_iso": res_iso,
            "res_ks": res_ks,
            "ks_theory": ks_theory,
            "ks_theory_coef": coef,
            "ks_theory_intercept": intercept,
            "ks_theory_r2": r2_lin,
            "iso_theory": iso_theory,
            "iso_theory_coef": coef_iso,
            "iso_theory_intercept": intercept_iso,
            "iso_theory_r2": r2_iso
        })

        fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        axs[0].set_xscale("log")
        axs[0].set_yscale("log")
        axs[0].set_xlabel("Pred_Iso (log scale)")
        axs[0].set_ylabel("Time_Iso_Mean (log scale)")
        axs[0].set_title(f"Iso: Time vs Predictor ({gt})")
        style_axes(axs[0])
        plot_loglog_with_fit(axs[0], sub["Pred_Iso"].values, sub["Time_Iso_Mean"].values,
                             res_iso, iso_color, marker="o", label="Iso data")
        axs[0].legend(frameon=True, facecolor="white", edgecolor="0.8")

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

        save_fig(fig, os.path.join(
            OUT_DIR, f"{gt}_predictor_scatter_loglog.png"))

        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.plot(sub["Nodes"], sub["Time_Iso_Mean"], marker="o", label="Iso",
                color=iso_color, linewidth=2, alpha=0.95)
        ax.plot(sub["Nodes"], sub["Time_KS_Mean"], marker="s", label="KS",
                color=ks_color, linewidth=2, alpha=0.95)

        if not math.isnan(coef_iso):
            predicted_iso = coef_iso * iso_theory + intercept_iso
            ax.plot(sub["Nodes"], predicted_iso, linestyle="--", linewidth=2.0,
                    label=f"Iso theory (fit), R2={r2_iso:.3f}", color=iso_color, alpha=0.85)

        if not math.isnan(coef):
            predicted = coef * ks_theory + intercept
            ax.plot(sub["Nodes"], predicted, linestyle="--", linewidth=2.0,
                    label=f"KS theory (fit), R2={r2_lin:.3f}", color=ks_color, alpha=0.85)

        ax.set_xlabel("Nodes")
        ax.set_ylabel("Time (s)")
        ax.set_title(f"Time vs Nodes ({gt})")
        style_axes(ax)
        ax.legend(frameon=True)
        save_fig(fig, os.path.join(OUT_DIR, f"{gt}_time_vs_nodes.png"))

        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.scatter(sub["Time_Iso_Mean"], sub["Time_KS_Mean"], marker="o", s=60,
                   color=topo_color, edgecolor='k', alpha=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Time_Iso_Mean (log)")
        ax.set_ylabel("Time_KS_Mean (log)")
        ax.set_title(f"Iso time vs KS time ({gt})")
        style_axes(ax)
        minv = np.nanmin([sub["Time_Iso_Mean"].replace(
            0, np.nan).min(), sub["Time_KS_Mean"].replace(0, np.nan).min()])
        maxv = np.nanmax([sub["Time_Iso_Mean"].max(),
                         sub["Time_KS_Mean"].max()])
        if np.isfinite(minv) and np.isfinite(maxv) and minv > 0 and maxv > 0:
            xs = np.logspace(math.log10(minv), math.log10(maxv), 100)
            ax.plot(xs, xs, linestyle="--", color="gray",
                    linewidth=1.0, zorder=2)
        save_fig(fig, os.path.join(OUT_DIR, f"{gt}_iso_vs_ks_scatter.png"))

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for item in master_data:
        gt = item["Graph_Type"]
        color = item["color"]
        res = item["res_iso"]
        plot_loglog_with_fit(ax, item["Pred_Iso"], item["Time_Iso"], res, color,
                             marker="o", label=f"{gt} Iso")
    ax.set_xlabel("Pred_Iso (log scale)")
    ax.set_ylabel("Time_Iso_Mean (log scale)")
    ax.set_title(
        "Master (Iso): Predictor vs Time for all topologies (log-log)")
    style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=True, ncol=2)
    save_fig(fig, os.path.join(
        OUT_DIR, "master_iso_predictor_scatter_loglog.png"))

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for item in master_data:
        gt = item["Graph_Type"]
        color = item["color"]
        res = item["res_ks"]
        plot_loglog_with_fit(ax, item["Pred_KS"], item["Time_KS"], res, color,
                             marker="s", label=f"{gt} KS")
    ax.set_xlabel("Pred_KS (log scale)")
    ax.set_ylabel("Time_KS_Mean (log scale)")
    ax.set_title("Master (KS): Predictor vs Time for all topologies (log-log)")
    style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), frameon=True, ncol=2)
    save_fig(fig, os.path.join(OUT_DIR, "master_ks_predictor_scatter_loglog.png"))

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "loglog_summary.csv"), index=False)

    slopes_df = pd.DataFrame(loglog_rows)
    slopes_df.to_csv(os.path.join(
        OUT_DIR, "loglog_slopes_full.csv"), index=False)

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
