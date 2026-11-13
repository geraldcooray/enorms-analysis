"""Generate simulated window-level e-norms data.

For every nerve and parameter, one simulated window estimate (mean and SD of
the normal distribution) is generated at each age at which a window of 100
measurements was located in the original data (data/window_ages.csv):

    mean_sim = f_mean(age) * (1 + e1),            e1 ~ N(0, rel_noise_mean)
    sd_sim   = sd_fit(age) * (1 + e2),            e2 ~ N(0, rel_noise_sd)
    sd_fit   = (f_upper(age) - f_lower(age)) / 4

where f_mean, f_upper (mean + 2 SD) and f_lower (mean - 2 SD) are the fitted
trajectories of Figures 1 and 2 (data/enorms_trajectories.csv), and the
relative noise levels equal the scatter of the original window estimates
around those trajectories.  The output has the same layout as the e-norms
output used in the study, so all other scripts run on it unchanged.

Usage:
    python scripts/make_simulated_data.py [--seed 2026]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import enorms_models as em

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default=ROOT / "data" / "simulated")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    traj = pd.read_csv(ROOT / "data" / "enorms_trajectories.csv")
    ages = pd.read_csv(ROOT / "data" / "window_ages.csv")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind in ("Motor", "Sensory"):
        rows = []
        for (internal, p), g in traj[traj.kind == kind].groupby(["nerve_internal", "parameter"], sort=False):
            c = {r.curve: r for r in g.itertuples()}
            model = c["mean"].model
            par = lambda r: [r.A, r.B, r.C, r.D, r.E]
            age_m = ages[(ages.kind == kind) & (ages.nerve_internal == internal)
                         & (ages.parameter == p)]["age_months"].to_numpy()
            t = age_m / 12
            f_mean = em.evaluate(model, par(c["mean"]), t)
            sd_fit = (em.evaluate(model, par(c["upper"]), t) - em.evaluate(model, par(c["lower"]), t)) / 4
            mean = f_mean * (1 + rng.normal(0, c["mean"].rel_noise_mean, t.size))
            sd = sd_fit * (1 + rng.normal(0, c["mean"].rel_noise_sd, t.size))
            sd = np.maximum(sd, 0.1 * sd_fit)          # keep SD positive
            for am, m, s in zip(age_m, mean, sd):
                rows.append({"dataset": f"{internal}_{kind}_{p}_{am:.1f} mo",
                             "Mean Age (in months) for cluster": am,
                             "Cluster size": em.CLUSTER_SIZE,
                             "mean": round(m, 4), "sd": round(s, 4)})
        f = out_dir / f"{kind}_enorms_windows_simulated.xlsx"
        pd.DataFrame(rows).to_excel(f, index=False)
        print(f"wrote {f} ({len(rows)} windows)")


if __name__ == "__main__":
    main()
