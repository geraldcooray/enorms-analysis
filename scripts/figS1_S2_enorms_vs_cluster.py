"""Supplementary Figures S1 and S2: e-norms vs clustering-based normative limits.

Black: e-norms mean (solid) and +/-2 SD (dashed), fitted to the window-level
e-norms estimates exactly as for Figures 1-2.
Blue: clustering-method mean (solid) and 5th-95th percentiles (dotted, shaded),
drawn from the stored bi-exponential constants in
data/cluster_fit_parameters.csv.

The clustering curves were obtained from the original patient-level data
(Cooray et al., 2026): per nerve, data ordered by age and split into windows of
50 studies; within each window all parameters were standardised, embedded with
t-SNE and split into two clusters with K-means; the cluster with the higher
mean amplitude was taken as normal, and the mean and 5th/95th percentiles of
each parameter were fitted against age with the same bi-exponential models.
Only the fitted constants are distributed.

Usage:
    python scripts/figS1_S2_enorms_vs_cluster.py \
        [--motor data/simulated/Motor_enorms_windows_simulated.xlsx] \
        [--sensory data/simulated/Sensory_enorms_windows_simulated.xlsx] \
        [--cluster data/cluster_fit_parameters.csv] [--out-dir figures]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import enorms_models as em

ROOT = Path(__file__).resolve().parents[1]

FIGS = {
    "Motor": dict(nerves=[("Medianus", "Median"), ("Ulnaris", "Ulnar"),
                          ("Tibialis", "Tibial"), ("Peroneus", "Peroneal")],
                  parameters=["Amp", "Onset", "CV"],
                  ylims={"Amp": (0, 20), "Onset": (0, 7), "CV": (0, 80)},
                  ylabels={"Amp": "Amplitude, mV", "Onset": "Onset / DML, ms",
                           "CV": "Conduction velocity, m/s"},
                  png="figS1_motor_enorms_vs_cluster.png",
                  title="Motor Nerve Responses: e-norms vs Cluster"),
    "Sensory": dict(nerves=[("Medianus", "Median"), ("DigPlantMed", "Medial Plantar"),
                            ("Suralis", "Sural"), ("Peroneus", "Peroneal")],
                    parameters=["Amp", "CV"],
                    ylims={"Amp": (0, 150), "CV": (0, 100)},
                    ylabels={"Amp": "Amplitude, μV", "CV": "Conduction velocity, m/s"},
                    png="figS2_sensory_enorms_vs_cluster.png",
                    title="Sensory Nerve Responses: e-norms vs Cluster"),
}


def cluster_curve(cl, kind, nerve, p, curve, x):
    r = cl[(cl.kind == kind) & (cl.nerve == nerve) & (cl.parameter == p) & (cl.curve == curve)].iloc[0]
    return em.evaluate(r.model, [r.A, r.B, r.C, r.D, r.E], x)


def cluster_range(cl, kind, nerve, p):
    r = cl[(cl.kind == kind) & (cl.nerve == nerve) & (cl.parameter == p)].iloc[0]
    return r.age_min_y, r.age_max_y


def make_figure(kind, data_file, cl, out_dir):
    cfg = FIGS[kind]
    df = em.load_windows(data_file)
    nerves, params = cfg["nerves"], cfg["parameters"]
    fig, axes = plt.subplots(len(nerves), len(params), figsize=(15, 16), squeeze=False)
    for r, (internal, disp) in enumerate(nerves):
        for c, p in enumerate(params):
            ax = axes[r, c]
            # e-norms
            try:
                x, _, fits = em.fit_trajectories(df, internal, kind, p)
                xf = np.linspace(x.min(), x.max(), 500)
                m = fits["model"]
                ax.plot(xf, em.evaluate(m, fits["mean"], xf), color="black", lw=2.3, label="e-norms mean")
                ax.plot(xf, em.evaluate(m, fits["upper"], xf), color="black", ls="--", lw=1.3, label="e-norms ±2SD")
                ax.plot(xf, em.evaluate(m, fits["lower"], xf), color="black", ls="--", lw=1.3)
            except RuntimeError as e:
                print(f"  e-norms fit failed {disp} {p}: {e}")
            # clustering (stored constants)
            a0, a1 = cluster_range(cl, kind, disp, p)
            xf = np.linspace(a0, a1, 500)
            mean, lo, hi = (cluster_curve(cl, kind, disp, p, k, xf) for k in ("mean", "lower", "upper"))
            ax.plot(xf, mean, color="blue", lw=2, label="Cluster mean")
            ax.plot(xf, hi, color="blue", ls=":", lw=1.7, label="Cluster p5–p95")
            ax.plot(xf, lo, color="blue", ls=":", lw=1.7)
            ax.fill_between(xf, lo, hi, color="blue", alpha=0.08)

            ax.set_title(f"{disp} {kind} {p}")
            ax.set_xlabel("Age, years")
            ax.set_xlim(0, 18)
            ax.set_ylim(cfg["ylims"][p])
            ax.set_ylabel(cfg["ylabels"][p])
            ax.grid(alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False)
    fig.suptitle(cfg["title"], fontsize=16, y=0.995)
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
    ap.add_argument("--cluster", default=ROOT / "data" / "cluster_fit_parameters.csv")
    ap.add_argument("--out-dir", default=ROOT / "figures")
    a = ap.parse_args()
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    cl = pd.read_csv(a.cluster)
    make_figure("Motor", a.motor, cl, a.out_dir)
    make_figure("Sensory", a.sensory, cl, a.out_dir)


if __name__ == "__main__":
    main()
