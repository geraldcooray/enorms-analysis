"""Shared models, curve fitting and data access for the e-norms analysis.

The bi-exponential models, starting values and bounds are those used to
produce Figures 1 and 2 and the Supplementary Tables of the manuscript.

General form (main text, Section 2.3):
    f(age) = A - |B| exp(-|C| age) + |D| exp(-|E| age)
  * Motor amplitude and CV : D = E = 0 (single exponential).
  * Motor onset latency    : A + B exp(-|C| age) + |D| exp(-|E| age), B free.
  * Sensory amplitude/CV   : full bi-exponential.
Age is in years.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

CLUSTER_SIZE = 100

MOTOR_NERVES = [("Medianus", "Median"), ("Ulnaris", "Ulnar"),
                ("Tibialis", "Tibial"), ("Peroneus", "Peroneal")]
MOTOR_PARAMETERS = ["Amp", "Onset", "CV"]

SENSORY_NERVES = [("Medianus", "Median"), ("Ulnaris", "Ulnar"),
                  ("DigPlantMed", "Medial Plantar"), ("Peroneus", "Peroneal"),
                  ("Suralis", "Sural")]
SENSORY_PARAMETERS = ["Amp", "CV"]


# ======================================================================
# Models
# ======================================================================
def motor_single(t, A, B_abs, C_abs):
    """Motor amplitude / CV: A - |B| exp(-|C| t)."""
    return A - B_abs * np.exp(-C_abs * t)


def motor_double(t, A, B, C_abs, D, E_abs):
    """Motor onset latency: A + B exp(-|C| t) + D exp(-|E| t), B unconstrained."""
    return A + B * np.exp(-C_abs * t) + D * np.exp(-E_abs * t)


def sensory_double(t, A, B_abs, C_abs, D, E_abs):
    """Sensory amplitude / CV: A - |B| exp(-|C| t) + |D| exp(-|E| t)."""
    return A - B_abs * np.exp(-C_abs * t) + D * np.exp(-E_abs * t)


MODELS = {"single": motor_single, "motor_double": motor_double,
          "sensory_double": sensory_double}


def model_name(kind, parameter):
    if kind == "Motor":
        return "motor_double" if parameter == "Onset" else "single"
    return "sensory_double"


def evaluate(model, params, t):
    """Evaluate a named model with a parameter vector (NaNs ignored)."""
    p = [v for v in params if np.isfinite(v)]
    return MODELS[model](np.asarray(t, float), *p)


# ======================================================================
# Fitting (starting values and bounds as used for the manuscript)
# ======================================================================
def _finite(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def fit_motor_single(x, y):
    x, y = _finite(x, y)
    popt, _ = curve_fit(motor_single, x, y,
                        p0=[max(y.max(), 0.01), abs(y.max() - y.min()), 0.05],
                        bounds=([0, 0, 0], [np.inf] * 3), maxfev=20000)
    return popt


def fit_motor_double(x, y):
    x, y = _finite(x, y)
    popt, _ = curve_fit(motor_double, x, y,
                        p0=[max(y[-1], 0.01), y.max() - y.min(), 0.02,
                            max(abs(y.max() - y.min()) / 2, 0.01), 0.3],
                        bounds=([0, -np.inf, 0, 0, 0], [np.inf] * 5),
                        maxfev=50000)
    return popt


def fit_sensory_double(x, y):
    x, y = _finite(x, y)
    popt, _ = curve_fit(sensory_double, x, y,
                        p0=[max(y[-1], 0.01), max(abs(y.max() - y.min()), 0.01), 0.05,
                            max(abs(y.max() - y.min()) / 2, 0.01), 0.1],
                        bounds=([0] * 5, [np.inf] * 5), maxfev=100000)
    return popt


FITTERS = {"single": fit_motor_single, "motor_double": fit_motor_double,
           "sensory_double": fit_sensory_double}


# ======================================================================
# Data access
# ======================================================================
def load_windows(path):
    """Window-level e-norms estimates (one row per window of 100 values).

    Required columns: 'dataset' ("<Nerve>_<Motor|Sensory>_<Param>_<age> mo"),
    'Cluster size', 'mean', 'sd'.
    """
    return pd.read_excel(path)


def get_data_for(df, nerve_internal, kind, parameter, cluster_size=CLUSTER_SIZE):
    """Return age (years), mean, mean + 2 SD and mean - 2 SD for one nerve/parameter."""
    s = df[df["dataset"].astype(str).str.contains(f"{nerve_internal}_{kind}_{parameter}",
                                                  case=False, na=False)
           & (df["Cluster size"] == cluster_size)].copy()
    s["age_months"] = s["dataset"].astype(str).str.extract(r"_(\d+(?:\.\d+)?)\s*mo")[0].astype(float)
    s = s.sort_values("age_months")
    x = s["age_months"].to_numpy() / 12
    mean, sd = s["mean"].to_numpy(), s["sd"].to_numpy()
    return x, mean, mean + 2 * sd, mean - 2 * sd


def fit_trajectories(df, nerve_internal, kind, parameter):
    """Fit the mean, upper (+2 SD) and lower (-2 SD) curves.

    Returns (x, data, fits) where data = {'mean','upper','lower'} arrays and
    fits = {'model': name, 'mean': popt, 'upper': popt, 'lower': popt}.
    """
    x, mean, upper, lower = get_data_for(df, nerve_internal, kind, parameter)
    name = model_name(kind, parameter)
    fit = FITTERS[name]
    data = {"mean": mean, "upper": upper, "lower": lower}
    fits = {"model": name}
    for k, y in data.items():
        fits[k] = fit(x, y)
    return x, data, fits
