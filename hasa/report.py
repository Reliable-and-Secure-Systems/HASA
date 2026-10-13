"""Tables and figures for runs over several seeds (used by the demo notebook).

`runs` maps a policy name ("Uniform", "HASA") to the list of histories returned by
run_policy, one per seed, in the same seed order for every policy.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

COLORS = {"Uniform": "#4B4A9A", "HASA": "#C33D6C"}
INK, MUTED, GRID = "#222238", "#6F7182", "#E6E9F2"
METRICS = [("test_mean", "Mean accuracy (%)"), ("test_worst", "Worst-client accuracy (%)"),
           ("test_p10", "P10 accuracy (%)")]


def final(runs, name, key, scale=100.0):
    """Last-round value of `key` for every seed."""
    return np.array([h[key][-1] for h in runs[name]]) * scale


def _style():
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "axes.titlecolor": INK, "axes.titlesize": 13})


def summary_table(runs, base="Uniform", new="HASA"):
    """Mean ± std over seeds, the paired gain, and a one-sided paired t-test (new > base)."""
    rows = []
    for key, label in METRICS + [("test_ppl", "Perplexity (lower is better)")]:
        scale = 1.0 if key == "test_ppl" else 100.0
        a, b = final(runs, base, key, scale), final(runs, new, key, scale)
        d = b - a
        better = d < 0 if key == "test_ppl" else d > 0
        alt = "less" if key == "test_ppl" else "greater"
        p = stats.ttest_rel(b, a, alternative=alt).pvalue if len(d) > 1 else float("nan")
        fmt = "{:.1f} ± {:.1f}" if key == "test_ppl" else "{:.2f} ± {:.2f}"
        rows.append({"": label, base: fmt.format(a.mean(), a.std()), new: fmt.format(b.mean(), b.std()),
                     "change": f"{d.mean():+.1f}" if key == "test_ppl" else f"{d.mean():+.2f} pp",
                     f"{new} better": f"{int(better.sum())} of {len(d)} seeds",
                     "p (paired t-test)": "< 0.0001" if p < 1e-4 else f"{p:.4f}"})
    return pd.DataFrame(rows).set_index("")


def plot_results(runs, seeds, base="Uniform", new="HASA"):
    """Left: the three client-level metrics, mean ± std over seeds. Right: mean accuracy seed by seed."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.4, 1]})

    rows = np.arange(len(METRICS))[::-1]
    for name, dy in ((base, 0.13), (new, -0.13)):
        vals = [final(runs, name, k) for k, _ in METRICS]
        m, s = np.array([v.mean() for v in vals]), np.array([v.std() for v in vals])
        ax1.errorbar(m, rows + dy, xerr=s, fmt="o", ms=9, color=COLORS[name], elinewidth=2, capsize=4,
                     label=name)
        for xi, si, yi in zip(m, s, rows + dy):
            ax1.text(xi + si + 0.08, yi, f"{xi:.2f}", va="center", fontsize=11, color=INK)
    ax1.set_yticks(rows, ["Mean", "Worst client", "P10"])
    ax1.set_ylim(-0.6, len(METRICS) - 0.4)
    ax1.set_xlabel("client test accuracy (%)")
    ax1.grid(axis="x", color=GRID)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, loc="lower right")
    ax1.set_title(f"Same compute budget, {len(seeds)} seeds (mean ± std)", loc="left")

    a, b = final(runs, base, "test_mean"), final(runs, new, "test_mean")
    for ya, yb in zip(a, b):
        ax2.plot([0, 1], [ya, yb], color=COLORS[new] if yb > ya else MUTED, lw=1.6, alpha=0.85, zorder=1)
    ax2.scatter(np.zeros(len(a)), a, s=55, color=COLORS[base], zorder=2)
    ax2.scatter(np.ones(len(b)), b, s=55, color=COLORS[new], zorder=2)
    ax2.set_xlim(-0.4, 1.4)
    ax2.set_xticks([0, 1], [base, new])
    ax2.set_ylabel("mean client accuracy (%)")
    ax2.grid(axis="y", color=GRID)
    ax2.set_axisbelow(True)
    ax2.set_title(f"Seed by seed: {new} higher in {int((b > a).sum())} of {len(a)}", loc="left")
    plt.tight_layout()
    plt.show()


def per_client_table(runs, client_names, widths, base="Uniform", new="HASA"):
    """Test accuracy of every client (mean ± std over seeds) and its gain under the new allocation."""
    a = np.array([h["per_client_test"][-1] for h in runs[base]]) * 100
    b = np.array([h["per_client_test"][-1] for h in runs[new]]) * 100
    return pd.DataFrame({"client": client_names,
                         f"{new} width (%)": np.round(100 * np.asarray(widths), 1),
                         f"{base} (%)": [f"{m:.2f} ± {s:.2f}" for m, s in zip(a.mean(0), a.std(0))],
                         f"{new} (%)": [f"{m:.2f} ± {s:.2f}" for m, s in zip(b.mean(0), b.std(0))],
                         "gain (pp)": np.round((b - a).mean(0), 2)}).set_index("client")


def plot_client_gains(table, new="HASA"):
    """Horizontal bars: average gain of every client, labelled with its subnet width."""
    _style()
    gains = table["gain (pp)"].to_numpy()
    labels = [f"{c} ({w:.0f}%)" for c, w in zip(table.index, table[f"{new} width (%)"])]
    y = np.arange(len(gains))[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.barh(y, gains, 0.6, color=[COLORS[new] if g >= 0 else MUTED for g in gains])
    for yi, g in zip(y, gains):
        ax.text(g + (0.04 if g >= 0 else -0.04), yi, f"{g:+.2f}", va="center",
                ha="left" if g >= 0 else "right", fontsize=11, color=INK)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, labels)
    span = max(abs(gains).max(), 0.5) * 1.35
    ax.set_xlim(-span, span)
    ax.set_xlabel(f"accuracy gain of {new} over uniform widths (pp)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(f"Gain per client ({new} width in brackets)", loc="left")
    plt.tight_layout()
    plt.show()
