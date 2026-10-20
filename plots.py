"""Figures for the report and demo (static PNGs).

Colours are the first three slots of a validated categorical palette
(blue, orange, aqua) in fixed order; text stays in neutral ink.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua (fixed order)


def _style(ax, xlabel, ylabel, title):
    ax.figure.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left")


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_ratio_vs_period(stats, t_used, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.fill_between(stats["period"], stats["p16"], stats["p84"], color=SERIES[0], alpha=0.18,
                    linewidth=0)
    ax.plot(stats["period"], stats["median"], color=SERIES[0], linewidth=2)
    ax.axvline(t_used, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(t_used, ax.get_ylim()[1], f"  T = {t_used:g} s used for modelling", color=INK2,
            fontsize=8, va="top")
    ax.set_xscale("log")
    _style(ax, "Period T (s)", "RotD100 / RotD50",
           "Directionality ratio by period (median, 16th-84th pct)")
    _save(fig, path)


def plot_ratio_hist(ratio, threshold, t_used, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.hist(ratio, bins=50, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
    ax.axvline(threshold, color=SERIES[1], linewidth=2)
    ax.text(threshold, ax.get_ylim()[1] * 0.95, f"  strong directionality\n  ratio >= {threshold:.2f}",
            color=INK2, fontsize=8, va="top")
    _style(ax, f"RotD100 / RotD50 at T = {t_used:g} s", "Number of records",
           "Distribution of the directionality ratio")
    _save(fig, path)


def plot_roc(y, oof, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=(0, (4, 3)), label="Chance")
    i = 0
    for name, prob in oof.items():
        if name.startswith("Baseline"):
            continue
        fpr, tpr, _ = roc_curve(y, prob)
        ax.plot(fpr, tpr, color=SERIES[i % 3], linewidth=2,
                label=f"{name} (AUC {roc_auc_score(y, prob):.2f})")
        i += 1
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    _style(ax, "False positive rate", "True positive rate",
           "ROC: detecting strong directionality")
    _save(fig, path)


def plot_pred_vs_actual(y_log, pred_log, model_name, r2, path):
    obs, pred = np.exp(y_log), np.exp(pred_log)
    lim = (min(obs.min(), pred.min()) * 0.98, np.percentile(np.r_[obs, pred], 99.5))
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(pred, obs, s=6, color=SERIES[0], alpha=0.25, linewidths=0)
    ax.plot(lim, lim, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    _style(ax, "Predicted RotD100 / RotD50", "Observed RotD100 / RotD50",
           f"{model_name} (R² = {r2:.2f})")
    _save(fig, path)


def plot_importance(imp, path, top=10):
    imp = imp[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.barh([r["feature"] for r in imp], [max(r["importance"], 0) for r in imp],
            color=SERIES[0], height=0.6)
    _style(ax, "Drop in R² when the feature is shuffled", "",
           "Which features drive the prediction")
    ax.grid(axis="y", visible=False)
    _save(fig, path)
