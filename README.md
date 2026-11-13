# e-norms analysis: paediatric nerve conduction normative values

Code accompanying the manuscript on age-continuous normative values for paediatric
nerve conduction studies estimated with e-norms. The scripts reproduce Figures 1–2,
Supplementary Figures S1–S2 and Supplementary Tables 1–9.

**No patient data are included.** The scripts run on *simulated* window-level data
generated from the fitted normative trajectories reported in the manuscript (see
[Simulated data](#simulated-data)). Results obtained with the simulated data are therefore
close to, but not identical with, those in the manuscript.

## Repository structure

```
scripts/
  enorms_models.py               bi-exponential models, curve fitting, data access (shared)
  make_simulated_data.py         generates data/simulated/*.xlsx from the fitted trajectories
  fig1_fig2_enorms.py            Figure 1 (motor) and Figure 2 (sensory)
  reference_tables.py            Supplementary Tables 1–9 (age-stratified reference values)
  figS1_S2_enorms_vs_cluster.py  Supplementary Figures S1–S2 (e-norms vs clustering method)
data/
  enorms_trajectories.csv        fitted e-norms curves (mean, mean ± 2 SD) and noise levels
  window_ages.csv                mean age of each window of 100 measurements
  cluster_fit_parameters.csv     fitted clustering-method curves (mean, 5th, 95th percentile)
  simulated/                     simulated window-level e-norms data (motor, sensory)
figures/                         output figures (generated from the simulated data)
results/                         output tables (generated from the simulated data)
```

## Installation and use

Python 3.11 was used.

```bash
pip install -r requirements.txt

python scripts/make_simulated_data.py         # optional: data/simulated/ is already provided
python scripts/fig1_fig2_enorms.py            # -> figures/fig1_*.png, figures/fig2_*.png
python scripts/reference_tables.py            # -> results/*_reference_tables.xlsx
python scripts/figS1_S2_enorms_vs_cluster.py  # -> figures/figS1_*.png, figures/figS2_*.png
```

Each script takes `--motor` / `--sensory` arguments, so it can be run on other
window-level e-norms output with the same layout (see below).

## Methods implemented

### Window-level e-norms estimates
For each nerve and parameter, the measurements were ordered by age and split into
consecutive windows of 100 values. Within each window, e-norms estimated the mean and
standard deviation (SD) of the normal distribution from the plateau of the sorted values,
which was located with a third-order polynomial fit. The scripts start from these
window-level estimates. Input files have one row per window, with these columns:

| column | content |
|---|---|
| `dataset` | `<Nerve>_<Motor\|Sensory>_<Parameter>_<age> mo`, e.g. `Medianus_Motor_Amp_12.3 mo` |
| `Cluster size` | number of measurements per window (100) |
| `mean`, `sd` | e-norms estimate of the mean and SD of the normal distribution |

Nerves: `Medianus`, `Ulnaris`, `Tibialis`, `Peroneus` (motor); `Medianus`, `Ulnaris`,
`DigPlantMed`, `Peroneus`, `Suralis` (sensory). Parameters: `Amp`, `Onset`, `CV`.

### Age-continuous curves
The window means, means + 2 SD and means − 2 SD were each fitted against age (years) with

    f(age) = A − |B|·exp(−|C|·age) + |D|·exp(−|E|·age)

* motor amplitude and conduction velocity: D = E = 0;
* motor onset latency: B unconstrained (initial dip followed by a rise);
* sensory amplitude and conduction velocity: full bi-exponential.

Fitting uses non-linear least squares (`scipy.optimize.curve_fit`) with the starting
values and bounds in `enorms_models.py`.

### Reference tables
For each age group, the fitted curves are evaluated at 200 equally spaced ages. The group
mean is the average of the mean curve, and the group SD is the average of
(upper − lower)/4. Values are given as mean (mean − 2 SD to mean + 2 SD).

### Comparison with the clustering method (S1–S2)
The clustering-based normative limits (Cooray et al., 2026) were derived from the
patient-level data as follows:
* data ordered by age and split into windows of 50 studies;
* t-SNE embedding of all standardised parameters, followed by K-means with k = 2;
* the cluster with the higher mean amplitude was taken as normal;
* the mean and the 5th and 95th percentiles of each parameter were fitted with the same
  bi-exponential models.

Only the fitted constants are distributed (`data/cluster_fit_parameters.csv`; columns
`model`, `A`–`E`, and the fitted age range in years). Some of these fits contain large,
nearly cancelling terms. They are stored at full precision and should be evaluated only
within `age_min_y`–`age_max_y`.

## Simulated data
`make_simulated_data.py` places one simulated window at each window age in the original
data (`data/window_ages.csv`):

    mean_sim = f_mean(age) · (1 + ε₁),   ε₁ ~ N(0, rel_noise_mean)
    sd_sim   = sd_fit(age) · (1 + ε₂),   ε₂ ~ N(0, rel_noise_sd),   sd_fit = (f_upper − f_lower)/4

* `f_mean`, `f_upper` and `f_lower` are the fitted Figure 1–2 trajectories
  (`data/enorms_trajectories.csv`).
* The relative noise levels equal the scatter of the original window estimates around
  those trajectories.
* The default random seed is 2026.

When run on the original window-level data, the scripts reproduce the published figures
and tables.

## Licence and citation
Please cite the accompanying manuscript when using this code.
