"""Supplementary Tables 1-9: age-stratified reference values.

The mean, mean + 2 SD and mean - 2 SD curves are fitted as for Figures 1-2.
Within each age group the fitted curves are evaluated at 200 equally spaced
ages; the group mean is the average of the fitted mean curve and the group SD
is the average of (upper - lower) / 4.  Values are reported as
    mean (mean - 2 SD to mean + 2 SD).

Usage:
    python scripts/reference_tables.py \
        [--motor data/simulated/Motor_enorms_windows_simulated.xlsx] \
        [--sensory data/simulated/Sensory_enorms_windows_simulated.xlsx] \
        [--out-dir results]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import enorms_models as em

ROOT = Path(__file__).resolve().parents[1]

AGE_GROUPS = [("0-3m", 0, 3 / 12), ("3-6m", 3 / 12, 6 / 12), ("6-9m", 6 / 12, 9 / 12),
              ("9-12m", 9 / 12, 1), ("12-15m", 1, 15 / 12), ("15-18m", 15 / 12, 18 / 12),
              ("18-24m", 18 / 12, 2), ("2-3y", 2, 3), ("3-4y", 3, 4), ("4-6y", 4, 6),
              ("6-10y", 6, 10), ("10-14y", 10, 14), ("14-18y", 14, 18)]

UNITS = {"Motor": {"Amp": "Amp (mV)", "Onset": "Onset (ms)", "CV": "CV (m/s)"},
         "Sensory": {"Amp": "Amp (µV)", "CV": "CV (m/s)"}}


def age_group_summary(fits, a0, a1, n_points=200):
    ages = np.linspace(a0, a1, n_points)
    m = fits["model"]
    mean = em.evaluate(m, fits["mean"], ages).mean()
    sd = ((em.evaluate(m, fits["upper"], ages) - em.evaluate(m, fits["lower"], ages)) / 4).mean()
    return mean, sd


def tables(kind, data_file):
    df = em.load_windows(data_file)
    nerves = em.MOTOR_NERVES if kind == "Motor" else em.SENSORY_NERVES
    params = em.MOTOR_PARAMETERS if kind == "Motor" else em.SENSORY_PARAMETERS
    out = {}
    for internal, display in nerves:
        t = pd.DataFrame({"Age group": [g[0] for g in AGE_GROUPS]})
        for p in params:
            _, _, fits = em.fit_trajectories(df, internal, kind, p)
            col = []
            for _, a0, a1 in AGE_GROUPS:
                mean, sd = age_group_summary(fits, a0, a1)
                mean, sd = round(mean, 2), round(sd, 2)   # limits from the rounded values
                col.append(f"{mean:.2f} ({mean - 2 * sd:.2f}–{mean + 2 * sd:.2f})")
            t[UNITS[kind][p]] = col
        out[display] = t
    return out


def main():
    sim = ROOT / "data" / "simulated"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motor", default=sim / "Motor_enorms_windows_simulated.xlsx")
    ap.add_argument("--sensory", default=sim / "Sensory_enorms_windows_simulated.xlsx")
    ap.add_argument("--out-dir", default=ROOT / "results")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind, f in (("Motor", a.motor), ("Sensory", a.sensory)):
        tabs = tables(kind, f)
        out = out_dir / f"{kind.lower()}_reference_tables.xlsx"
        with pd.ExcelWriter(out) as w:
            for name, t in tabs.items():
                t.to_excel(w, sheet_name=name, index=False)
                print(f"\n{kind} – {name}\n{t.to_string(index=False)}")
        print("\nwrote", out)


if __name__ == "__main__":
    main()
