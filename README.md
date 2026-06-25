# Minmax Median-of-Means for Robust Conformal Calibration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Reproducible experiments for **"Minmax Median-of-Means for Robust Conformal Calibration"** (Statistics & Computing, under review).

MOM-conformal replaces the standard empirical quantile in split conformal prediction with a minmax median-of-means (MOM) estimator, making calibration robust to adversarial contamination of the held-out scores. This repository provides the full experimental pipeline.

---

## Quick start

```bash
git clone https://github.com/<your-username>/mom-conformal.git
cd mom-conformal
pip install numpy scipy matplotlib scikit-learn
cd experiments

# Option A: generate figures from pre-computed results (instant)
python make_figures.py

# Option B: re-run all experiments, then generate figures (≈30 min)
python run_experiments.py
python make_figures.py
```

Figures are written to `../output/`.

| Package | Version tested | Purpose |
|---------|------|---------|
| `numpy` | 1.24 | Core arrays, quantiles |
| `scipy` | 1.11 | Student-$t$ and log-normal distributions |
| `matplotlib` | 3.7 | Figures (300–400 dpi) |
| `scikit-learn` | 1.3 | Real-data models, data fetching |

---

## Repository structure

```
.
├── README.md
├── .gitignore
└── experiments/
    ├── run_experiments.py         # Full experiment suite
    ├── make_figures.py            # Nature-quality figure generator
    └── results/                   # Pre-computed results (8 .npz)
        ├── atlas_fig1.npz
        ├── mixed_contam.npz
        ├── multi_alpha.npz
        ├── extra_realdata.npz
        ├── contam_strength.npz
        ├── conditional_coverage.npz
        ├── partition_strategy.npz
        └── aci_baseline.npz
```

---

## Experiments

All experiments use **fixed random seeds** (`SEED = 42`) with per-replication, per-configuration offsets. Both scripts are fully self-contained; `run_experiments.py` writes `.npz` files to `results/`, and `make_figures.py` reads them and produces PDF figures in `../output/`.

### Methods compared

| Method | Oracle | Notes |
|--------|:------:|-------|
| Split conformal | — | Ceiling of $(1-\alpha)(n+1)$ quantile |
| CQR (Romano et al., 2019) | — | Conformalized quantile regression |
| Trimmed conformal | $\varepsilon$ | Trim top/bottom $100\varepsilon\%$ |
| Winsorized conformal | $\varepsilon$ | Winsorize top $100\varepsilon\%$ |
| MOM-conformal $(K)$ | — | Block-median-of-quantiles, $K \in \{10,20,40,80\}$ |
| Grid MOM | — | Minmax-MOM grid solver (Algorithm 1 in paper) |
| ACI (Gibbs \& Candès, 2021) | — | Supplement only |

### Experiment index

#### Experiment 1: Synthetic Robustness Atlas
- 4 distributions ($\mathcal{N}(0,1)$, $t_3$, $t_{1.5}$, $\log\mathcal{N}(0,1)$)
- High-value ($M = +8$/+15) and low-value ($M = -6$/−10) contamination
- $\varepsilon \in \{0,0.02,0.05,0.08,0.10,0.15,0.20\}$, $K\in\{10,20,40,80\}$
- $n_\text{cal}=2000$, $n_\text{test}=500$, $N=300$

#### Experiment 2: Mixed-direction contamination
- $\varepsilon=0.10$, 5 high:low mix ratios: $\{100/0, 75/25, 50/50, 25/75, 0/100\}$
- Gaussian and $t_3$, $n=2000$, $N=500$

#### Experiment 3: Multi-$\alpha$ coverage
- $\alpha \in \{0.05, 0.10, 0.20\}$ ($\tau\in\{0.95,0.90,0.80\}$)
- Gaussian and $t_3$, high and low contamination
- $n=2000$, $N=300$

#### Experiment 4: Real-data benchmarks
| Dataset | $n$ | Task | Model |
|---------|-----|------|-------|
| Wine Quality | 1,599 | Regression | RandomForest (200 trees) |
| MNIST (subset) | 10,000 | Classification | LogisticRegression |

- $\varepsilon\in\{0,0.05,0.10,0.15,0.20\}$, $N=100$
- Plus CA Housing ($n=20{,}640$) and Diabetes ($n=442$) in the paper.

#### Experiment 5: Contamination strength sensitivity
- $\varepsilon=0.10$, vary contamination magnitude $M$
- $M\in\{2,4,6,8,10,15,20\}$ (Gaussian), $M\in\{3,5,10,15,20,25,30\}$ ($t_3$)
- $n=2000$, $N=200$

#### Experiment 6: Conditional coverage
- 5 score-quantile bins (Q0–Q4), $\varepsilon\in\{0,0.05,0.10\}$
- High-value contamination, Gaussian scores
- $n_\text{cal}=2000$, $n_\text{test}=2000$, $N=200$

#### Experiment 7: Partition strategy ablation
- Random vs. stratified (interleave sorted) vs. contiguous (consecutive blocks)
- $\varepsilon\in\{0,0.05,0.10\}$, $K\in\{10,20,40\}$
- Gaussian scores, $n=2000$, $N=200$

### Key results

| Scenario | Split | MOM $(K=40)$ | Trimmed (oracle) |
|----------|:-----:|:------------:|:----------------:|
| $\varepsilon=0.10$ high (Gaussian), coverage | 1.000 | 0.995 | 0.912 |
| $\varepsilon=0.10$ high (Gaussian), width inflation | $6.2\times$ | $2.1\times$ | $1.1\times$ |
| $\varepsilon=0.10$ low (Gaussian), coverage | 0.890 | 0.878 | 0.800 |
| $\varepsilon=0$, width vs. split | $1.00\times$ | $0.95\times$ | $1.00\times$ |
| Runtime at $n = 5\times10^5$ | 12 ms | 6 ms | — |

---

## Output files

| `.npz` | Contents | Size |
|--------|----------|------|
| `atlas_fig1.npz` | 4×4×8×7: dists × metrics × methods × ε | 560 KB |
| `mixed_contam.npz` | 5 mix ratios × 7 methods × 500 reps | 560 KB |
| `multi_alpha.npz` | 3 α × 3 ε × 5 methods × 300 reps | 440 KB |
| `extra_realdata.npz` | Wine Quality + MNIST coverage/width data | 3 KB |
| `contam_strength.npz` | 7 $M$ values × 5 methods × 200 reps | 10 KB |
| `conditional_coverage.npz` | 5 bins × 5 methods × 200 reps, 3 ε | 80 KB |
| `partition_strategy.npz` | 3 strategies × 3 $K$ × 3 ε × 200 reps | 10 KB |
| `aci_baseline.npz` | ACI baseline coverage (supplement) | 2 KB |

---

## Hardware

All experiments were run on an Intel Core i7-13700K (3.4 GHz, 16 cores, 32 GB RAM), Python 3.11, single-core wall-clock timing.

---

## Citation

```bibtex
@article{momconformal2025,
  title   = {Minmax {M}edian-of-{M}eans for Robust Conformal Calibration},
  author  = {Author, First and Author, Second},
  journal = {Statistics and Computing},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT. See [LICENSE](LICENSE) for details.
