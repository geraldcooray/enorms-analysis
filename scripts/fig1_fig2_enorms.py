"""Figures 1 and 2: age-continuous normative curves for motor and sensory nerves.

For each nerve and parameter, the e-norms window estimates (mean, mean + 2 SD,
mean - 2 SD; one per window of 100 measurements ordered by age) are plotted
against age, and a bi-exponential curve is fitted to each of the three series
(see enorms_models.py).

Usage:
    python scripts/fig1_fig2_enorms.py \
        [--motor data/simulated/Motor_enorms_windows_simulated.xlsx] \
        [--sensory data/simulated/Sensory_enorms_windows_simulated.xlsx] \
        [--out-dir figures]
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import enorms_models as em

ROOT = Path(__file__).resolve().parents[1]

# Common x-axis for both figures (5% margin around the median motor amplitude windows)
XLIM = (-0.80, 18.44)
XTICKS = np.arange(0, 17.6, 2.5)

FIGS = {
    "Motor": dict(nerves=em.MOTOR_NERVES, parameters=em.MOTOR_PARAMETERS, figsize=(15, 16),
                  ylims={"Amp": (0, 14), "Onset": (0, 5.5), "CV": (0, 70)},
                  ylabels={"Amp": "Amplitude, mV", "Onset": "Onset, ms",
                           "CV": "Conduction velocity, m/s"},
                  min_points=4, png="fig1_motor_enorms.png"),
    "Sensory": dict(nerves=em.SENSORY_NERVES, parameters=em.SENSORY_PARAMETERS, figsize=(11, 18),
                    ylims={"Amp": (0, 80), "CV": (0, 75)},
                    ylabels={"Amp": "Amplitude, µV", "CV": "Conduction velocity, m/s"},
                    min_points=6, png="fig2_sensory_enorms.png"),
}


def make_figure(kind, data_file, out_dir):
    cfg = FIGS[kind]
    df = em.load_windows(data_file)
    nerves, params = cfg["nerves"], cfg["parameters"]
    fig, axes = plt.subplots(len(nerves), len(params), figsize=cfg["figsize"], squeeze=False)

    for i, (internal, display) in enumerate(nerves):
        for j, p in enumerate(params):
            ax = axes[i, j]
            x, mean, upper, lower = em.get_data_for(df, internal, kind, p)
            if len(x) < cfg["min_points"]:
                ax.set_title(f"{display} {kind} {p}: insufficient data")
                ax.axis("off")
                continue

            ax.scatter(x, mean, s=25, label="Mean")
            ax.scatter(x, upper, s=12, alpha=0.4, label="+2 SD")
            ax.scatter(x, lower, s=12, alpha=0.4, label="-2 SD")
            try:
                _, _, fits = em.fit_trajectories(df, internal, kind, p)
                x_fit = np.linspace(x.min(), x.max(), 500)
                for k, ls, lw in (("mean", "-", 2), ("upper", "--", 1.5), ("lower", "--", 1.5)):
                    ax.plot(x_fit, em.evaluate(fits["model"], fits[k], x_fit), linestyle=ls, linewidth=lw)
            except RuntimeError as e:
                ax.text(0.05, 0.95, f"Fit failed:\n{e}", transform=ax.transAxes, va="top", fontsize=8)

            ax.set_title(f"{display} {kind} {p}")
            ax.set_xlabel("Age, years")
            ax.set_ylabel(cfg["ylabels"][p])
            ax.set_ylim(cfg["ylims"][p])
            ax.set_xlim(XLIM)
            ax.set_xticks(XTICKS)
            ax.grid(alpha=0.3)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=3, frameon=False)
    fig.suptitle(f"{kind} Nerve Responses, Cluster Size {em.CLUSTER_SIZE}", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(out_dir) / cfg["png"]
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    sim = ROOT / "data" / "simulated"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motor", default=sim / "Motor_enorms_windows_simulated.xlsx")
    ap.add_argument("--sensory", default=sim / "Sensory_enorms_windows_simulated.xlsx")
    ap.add_argument("--out-dir", default=ROOT / "figures")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    make_figure("Motor", a.motor, a.out_dir)
    make_figure("Sensory", a.sensory, a.out_dir)


if __name__ == "__main__":
    main()
