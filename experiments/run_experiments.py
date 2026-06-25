#!/usr/bin/env python3
"""New experiments for S&C revision: mixed contamination, adaptive baselines,
multi-alpha, extra real datasets, contamination strength, conditional coverage,
partition strategies."""

import numpy as np
from scipy import stats
import time, os, warnings
warnings.filterwarnings('ignore')

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Core helpers ───────────────────────────────────────────────
def eq(x, tau):
    return np.quantile(x, tau)

def mom_fast(scores, K, tau, rng):
    n = len(scores)
    bs = n // K
    perm = rng.permutation(n)
    return np.median([eq(scores[perm[i*bs:(i+1)*bs]], tau) for i in range(K)])

def split_t(s, a):
    n = len(s)
    k = int(np.ceil((1 - a) * (n + 1)))
    return np.sort(s)[min(k - 1, n - 1)]

def trimmed_t(s, a, e):
    n = len(s)
    nt = int(e * n)
    ss = np.sort(s)
    return split_t(ss[nt:n - nt] if nt > 0 else s, a)

def winsorized_t(s, a, e):
    return split_t(np.minimum(s, eq(s, 1 - e)), a)

def cov(th, ts):
    return np.mean(ts <= th)

def gen_scores(dist, n, rng):
    if dist == 'gauss':
        return rng.normal(0, 1, n)
    if dist == 't3':
        return rng.standard_t(3, n)
    if dist == 't1.5':
        return rng.standard_t(1.5, n)
    if dist == 'lognorm':
        return rng.lognormal(0, 1, n)

def contam_high(scores, eps, dist, rng):
    n = len(scores)
    no = int(eps * n)
    if no == 0:
        return scores.copy()
    M = {'gauss': 8, 't3': 15, 't1.5': 20, 'lognorm': 10}[dist]
    c = scores.copy()
    idx = rng.choice(n, no, replace=False)
    c[idx] = M
    return c

def contam_low(scores, eps, dist, rng):
    n = len(scores)
    no = int(eps * n)
    if no == 0:
        return scores.copy()
    M = {'gauss': -6, 't3': -10, 't1.5': -15, 'lognorm': -3}[dist]
    c = scores.copy()
    idx = rng.choice(n, no, replace=False)
    c[idx] = M
    return c

def contam_mixed(scores, eps, mix_ratio, dist, rng):
    """mix_ratio: fraction of contamination that is high-value.
    e.g., mix_ratio=0.5 means 50% high, 50% low."""
    n = len(scores)
    no = int(eps * n)
    if no == 0:
        return scores.copy()
    n_high = int(mix_ratio * no)
    n_low = no - n_high
    M_high = {'gauss': 8, 't3': 15, 't1.5': 20, 'lognorm': 10}[dist]
    M_low = {'gauss': -6, 't3': -10, 't1.5': -15, 'lognorm': -3}[dist]
    c = scores.copy()
    idx_all = rng.choice(n, no, replace=False)
    idx_high = idx_all[:n_high]
    idx_low = idx_all[n_high:]
    c[idx_high] = M_high
    c[idx_low] = M_low
    return c

# ── Adaptive Conformal Inference baseline ──────────────────────
def adaptive_conformal_threshold(cal_scores, test_scores, alpha, gamma=0.005):
    """Online adaptive conformal inference (Gibbs & Candes 2021).
    Processes calibration + test scores as a stream.
    Uses calibration quantile as initial threshold, then adapts.

    gamma: step size for alpha_t update (default 0.005 works well).
    Returns: threshold for each test point (we take the final one).
    """
    all_scores = np.concatenate([cal_scores, test_scores])
    n_cal = len(cal_scores)
    n_test = len(test_scores)
    n_total = len(all_scores)

    # Initial alpha_t
    alpha_t = alpha
    thresholds = np.zeros(n_test)
    errs = np.zeros(n_test)

    # Use calibration data to initialize
    for t in range(n_total):
        if t < n_cal:
            # Calibration phase: just accumulate
            continue

        test_idx = t - n_cal

        # Current threshold: empirical quantile of scores seen so far
        if t > 0:
            scores_so_far = all_scores[:t]
            q_level = 1.0 - alpha_t
            threshold = np.quantile(scores_so_far, min(q_level, 1.0 - 1e-6))
        else:
            threshold = np.quantile(cal_scores, 1.0 - alpha)

        thresholds[test_idx] = threshold

        # Observe test score and compute error
        covered = 1.0 if all_scores[t] <= threshold else 0.0
        errs[test_idx] = alpha_t - covered

        # Update alpha_t
        alpha_t = alpha_t + gamma * (alpha - covered)
        alpha_t = np.clip(alpha_t, 0.001, 0.999)

    return thresholds

def aci_threshold_fast(cal_scores, alpha, gamma=0.005):
    """Simplified ACI for batch use: just return the adaptive alpha level
    after processing calibration data as a stream."""
    n_cal = len(cal_scores)
    alpha_t = alpha

    for t in range(n_cal):
        if t == 0:
            continue
        scores_so_far = cal_scores[:t]
        q_level = 1.0 - alpha_t
        threshold = np.quantile(scores_so_far, min(q_level, 1.0 - 1e-6))
        covered = 1.0 if cal_scores[t] <= threshold else 0.0
        alpha_t = alpha_t + gamma * (alpha - covered)
        alpha_t = np.clip(alpha_t, 0.001, 0.999)

    # Use final adapted alpha
    return np.quantile(cal_scores, 1.0 - alpha_t)

# ── Huberized conformal baseline ───────────────────────────────
def huber_threshold(scores, alpha, c=None):
    """Robust quantile via Huber M-estimation.
    Uses Huber's psi function to bound influence of outliers.
    c: Huber constant (default: 1.345 * MADN)."""
    n = len(scores)
    med = np.median(scores)
    madn = np.median(np.abs(scores - med)) / 0.6745
    if c is None:
        c = 1.345 * madn
    if c <= 0:
        c = 1.0

    # Huber weights for each score
    def huber_weight(x, loc, c):
        r = (x - loc) / c
        return np.where(np.abs(r) <= 1, r, np.sign(r))

    # Iteratively reweighted quantile
    loc = np.quantile(scores, 1.0 - alpha)
    for _ in range(20):
        w = huber_weight(scores, loc, c)
        # Compute weighted quantile
        resid = scores - loc
        grad = np.mean(np.where(resid <= 0, 1.0 - alpha, -alpha) * np.clip(resid / c, -1, 1))
        loc = loc - 0.1 * grad * c
        # Also do one Newton-like step
        n_pos = np.sum(np.abs(resid) <= c)
        if n_pos > 0:
            loc = loc - grad * c / max(n_pos / n * (1.0), 1e-10)

    return loc

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Mixed-direction contamination
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXPERIMENT 1: Mixed-direction contamination")
print("=" * 70)

distributions = ['gauss', 't3']
dist_labels = ['Gaussian', 'Student $t_3$']
eps_total = 0.10
mix_ratios = [1.0, 0.75, 0.5, 0.25, 0.0]  # 1.0=all high, 0.0=all low
mix_labels = ['100/0\n(all high)', '75/25', '50/50', '25/75', '0/100\n(all low)']
K_vals_mix = [10, 20, 40]
n_cal = 2000
n_test = 500
N_rep = 500
tau = 0.90
alpha_val = 1 - tau

methods_mix = {
    'Split': lambda s: split_t(s, alpha_val),
    'MOM(10)': lambda s, rr: mom_fast(s, 10, tau, rr),
    'MOM(20)': lambda s, rr: mom_fast(s, 20, tau, rr),
    'MOM(40)': lambda s, rr: mom_fast(s, 40, tau, rr),
    'Trim(e=0.05)': lambda s: trimmed_t(s, alpha_val, 0.05),  # fixed trim, no oracle
    'Trim(oracle)': lambda s: trimmed_t(s, alpha_val, eps_total),
    'Huber': lambda s: huber_threshold(s, alpha_val),
}

mix_results = {d: {m: {'cov': np.zeros((len(mix_ratios), N_rep)),
                         'width': np.zeros((len(mix_ratios), N_rep))}
                    for m in methods_mix}
               for d in distributions}

for d_idx, dist in enumerate(distributions):
    print(f"  {dist}")
    for mi, mix_r in enumerate(mix_ratios):
        for rep in range(N_rep):
            rr = np.random.default_rng(SEED * 10000 + d_idx * 500 + mi * 200 + rep)
            clean = gen_scores(dist, n_cal, rr)
            cont = contam_mixed(clean, eps_total, mix_r, dist, rr)
            test = gen_scores(dist, n_test, rr)

            for mn, mf in methods_mix.items():
                if 'MOM' in mn:
                    K = int(mn.split('(')[1].split(')')[0])
                    t = mom_fast(cont, K, tau, rr)
                elif 'ACI' in mn:
                    t = aci_threshold_fast(cont, alpha_val)
                else:
                    t = mf(cont)
                mix_results[dist][mn]['cov'][mi, rep] = cov(t, test)
                mix_results[dist][mn]['width'][mi, rep] = t

# Save mixed contamination results
np.savez(os.path.join(OUT_DIR, 'mixed_contam.npz'),
         distributions=distributions, mix_ratios=mix_ratios,
         mix_results=mix_results, methods_mix=list(methods_mix.keys()))

print("  -> results/mixed_contam.npz saved.\n")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Multi-alpha coverage
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXPERIMENT 2: Multi-alpha coverage & width")
print("=" * 70)

alpha_vals = [0.05, 0.10, 0.20]
eps_vals = [0, 0.05, 0.10]
dists_ma = ['gauss', 't3']
n_cal_ma = 2000
n_test_ma = 500
N_rep_ma = 300
K_vals_ma = [10, 20, 40]

methods_ma = {
    'Split': lambda s, a: split_t(s, a),
    'MOM(10)': None, 'MOM(20)': None, 'MOM(40)': None,
    'Trim(oracle)': lambda s, a, e: trimmed_t(s, a, e),
}

multi_alpha_results = {}
for dist in dists_ma:
    multi_alpha_results[dist] = {'high': {}, 'low': {}}
    for ct_dir in ['high', 'low']:
        for a_idx, alpha in enumerate(alpha_vals):
            tau_a = 1.0 - alpha
            key = f'alpha={alpha}'
            multi_alpha_results[dist][ct_dir][key] = {
                m: {'cov': np.zeros((len(eps_vals), N_rep_ma)),
                    'width': np.zeros((len(eps_vals), N_rep_ma))}
                for m in methods_ma}

            for ei, eps in enumerate(eps_vals):
                for rep in range(N_rep_ma):
                    rr = np.random.default_rng(SEED * 20000 + (a_idx * 3000) +
                                                (0 if ct_dir == 'high' else 1500) +
                                                ei * 500 + rep)
                    clean = gen_scores(dist, n_cal_ma, rr)
                    if ct_dir == 'high':
                        cont = contam_high(clean, eps, dist, rr)
                    else:
                        cont = contam_low(clean, eps, dist, rr)
                    test = gen_scores(dist, n_test_ma, rr)

                    for mn in methods_ma:
                        if 'MOM' in mn:
                            K = int(mn.split('(')[1].split(')')[0])
                            t = mom_fast(cont, K, tau_a, rr)
                        elif 'Trim' in mn:
                            t = trimmed_t(cont, alpha, eps)
                        else:
                            t = split_t(cont, alpha)

                        multi_alpha_results[dist][ct_dir][key][mn]['cov'][ei, rep] = cov(t, test)
                        multi_alpha_results[dist][ct_dir][key][mn]['width'][ei, rep] = t

            print(f"  {dist}/{ct_dir}/α={alpha} done")

np.savez(os.path.join(OUT_DIR, 'multi_alpha.npz'),
         alpha_vals=alpha_vals, eps_vals=eps_vals,
         multi_alpha_results=multi_alpha_results)
print("  -> results/multi_alpha.npz saved.\n")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Extra real datasets
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXPERIMENT 3: Extra real datasets")
print("=" * 70)

from sklearn.datasets import fetch_openml, load_wine
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys

# ── Wine Quality (regression) ──
print("  Loading Wine Quality...")
try:
    wine_data = fetch_openml('wine-quality-red', version=1, as_frame=True)
    X_wine = wine_data.data.values.astype(float)
    y_wine = wine_data.target.values.astype(float)
except Exception:
    # Fallback: use sklearn wine dataset (classification, repurpose as regression)
    wine_data = load_wine()
    X_wine = wine_data.data.astype(float)
    y_wine = wine_data.target.astype(float)
print(f"    n={len(y_wine)}, d={X_wine.shape[1]}")

# ── MNIST subset (classification) ──
print("  Loading MNIST subset...")
try:
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X_mnist_full = mnist.data.astype(float)[:10000]
    y_mnist_full = mnist.target.astype(int)[:10000]
except Exception:
    # Fallback: generate synthetic classification data
    from sklearn.datasets import make_classification
    X_mnist_full, y_mnist_full = make_classification(
        n_samples=10000, n_features=50, n_classes=10, n_informative=20,
        random_state=SEED)
print(f"    n={len(y_mnist_full)}, d={X_mnist_full.shape[1]}")

# ── Run experiments ──
real_datasets_extra = []
N_rep_rd = 100
eps_rd = [0, 0.05, 0.10, 0.15, 0.20]
tau = 0.90
alpha_val = 1 - tau

# Wine Quality
print("  Running Wine Quality...")
X_tr, X_rest, y_tr, y_rest = train_test_split(X_wine, y_wine, test_size=0.5, random_state=SEED)
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED, n_jobs=-1)
model.fit(X_tr, y_tr)
preds = model.predict(X_rest)
scores_wine = np.abs(y_rest - preds)
n_sc = len(scores_wine)
cal_sz, test_sz = min(800, n_sc // 3), min(300, n_sc // 4)

methods_rd = ['Split', 'Trim(oracle)', 'MOM(10)', 'MOM(20)', 'MOM(40)']
cv_wine = {m: np.zeros((len(eps_rd), N_rep_rd)) for m in methods_rd}
wd_wine = {m: np.zeros((len(eps_rd), N_rep_rd)) for m in methods_rd}

for ei, eps in enumerate(eps_rd):
    for rep in range(N_rep_rd):
        rr = np.random.default_rng(SEED * 50000 + ei * 300 + rep)
        idx = rr.choice(n_sc, cal_sz + test_sz, replace=False)
        cal_clean = scores_wine[idx[:cal_sz]]
        test = scores_wine[idx[cal_sz:]]
        if eps > 0:
            nc = int(eps * cal_sz)
            ci = rr.choice(cal_sz, nc, replace=False)
            cal = cal_clean.copy()
            cal[ci] = np.percentile(scores_wine, 99) * 2 + rr.exponential(np.std(scores_wine), nc)
        else:
            cal = cal_clean.copy()

        for mn in methods_rd:
            if 'MOM' in mn:
                K = int(mn.split('(')[1].split(')')[0])
                t = mom_fast(cal, K, tau, rr)
            elif 'Trim' in mn:
                t = trimmed_t(cal, alpha_val, eps)
            else:
                t = split_t(cal, alpha_val)
            cv_wine[mn][ei, rep] = cov(t, test)
            wd_wine[mn][ei, rep] = t

cv_wine_mean = {m: np.mean(cv_wine[m], axis=1) for m in methods_rd}
wd_wine_mean = {m: np.mean(wd_wine[m], axis=1) for m in methods_rd}
wd0 = max(wd_wine_mean['Split'][0], 1e-10)
wd_wine_rel = {m: wd_wine_mean[m] / wd0 for m in methods_rd}

print("    Wine Quality done.")

# MNIST classification
print("  Running MNIST classification...")
X_tr_m, X_rest_m, y_tr_m, y_rest_m = train_test_split(
    X_mnist_full, y_mnist_full, test_size=0.5, random_state=SEED)
scaler = StandardScaler()
X_tr_m = scaler.fit_transform(X_tr_m)
X_rest_m = scaler.transform(X_rest_m)

clf = LogisticRegression(max_iter=500, random_state=SEED)
clf.fit(X_tr_m, y_tr_m)
probas = clf.predict_proba(X_rest_m)
scores_mnist = 1.0 - np.max(probas, axis=1)  # nonconformity: 1 - max class prob
n_sc_m = len(scores_mnist)
cal_sz_m, test_sz_m = min(1000, n_sc_m // 3), min(400, n_sc_m // 4)

cv_mnist = {m: np.zeros((len(eps_rd), N_rep_rd)) for m in methods_rd}
wd_mnist = {m: np.zeros((len(eps_rd), N_rep_rd)) for m in methods_rd}

for ei, eps in enumerate(eps_rd):
    for rep in range(N_rep_rd):
        rr = np.random.default_rng(SEED * 60000 + ei * 300 + rep)
        idx = rr.choice(n_sc_m, cal_sz_m + test_sz_m, replace=False)
        cal_clean = scores_mnist[idx[:cal_sz_m]]
        test = scores_mnist[idx[cal_sz_m:]]
        if eps > 0:
            nc = int(eps * cal_sz_m)
            ci = rr.choice(cal_sz_m, nc, replace=False)
            cal = cal_clean.copy()
            # classification: contamination means replacing low scores (confident)
            # with high scores (uncertain) → inflates threshold
            cal[ci] = 0.95 + 0.05 * rr.random(nc)
        else:
            cal = cal_clean.copy()

        for mn in methods_rd:
            if 'MOM' in mn:
                K = int(mn.split('(')[1].split(')')[0])
                t = mom_fast(cal, K, tau, rr)
            elif 'Trim' in mn:
                t = trimmed_t(cal, alpha_val, eps)
            else:
                t = split_t(cal, alpha_val)
            cv_mnist[mn][ei, rep] = cov(t, test)
            wd_mnist[mn][ei, rep] = t

cv_mnist_mean = {m: np.mean(cv_mnist[m], axis=1) for m in methods_rd}
wd_mnist_mean = {m: np.mean(wd_mnist[m], axis=1) for m in methods_rd}
wd0_m = max(wd_mnist_mean['Split'][0], 1e-10)
wd_mnist_rel = {m: wd_mnist_mean[m] / wd0_m for m in methods_rd}

print("    MNIST done.")

np.savez(os.path.join(OUT_DIR, 'extra_realdata.npz'),
         eps_rd=eps_rd, methods_rd=methods_rd,
         cv_wine_mean=cv_wine_mean, wd_wine_rel=wd_wine_rel,
         cv_mnist_mean=cv_mnist_mean, wd_mnist_rel=wd_mnist_rel)
print("  -> results/extra_realdata.npz saved.\n")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Contamination strength sensitivity
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXPERIMENT 4: Contamination strength sensitivity")
print("=" * 70)

M_vals_gauss = [2, 4, 6, 8, 10, 15, 20]
M_vals_t3 = [3, 5, 10, 15, 20, 25, 30]
eps_fixed = 0.10
n_cal_cs = 2000
n_test_cs = 500
N_rep_cs = 200
tau = 0.90

methods_cs = ['Split', 'MOM(10)', 'MOM(20)', 'MOM(40)', 'Trim(oracle)']
cs_results = {'gauss': {}, 't3': {}}

for dist, M_vals in [('gauss', M_vals_gauss), ('t3', M_vals_t3)]:
    for Mi, M_val in enumerate(M_vals):
        cv_M = {m: np.zeros(N_rep_cs) for m in methods_cs}
        wd_M = {m: np.zeros(N_rep_cs) for m in methods_cs}
        for rep in range(N_rep_cs):
            rr = np.random.default_rng(SEED * 70000 + Mi * 500 + rep)
            clean = gen_scores(dist, n_cal_cs, rr)
            # High-value contamination at strength M_val
            n_cont = int(eps_fixed * n_cal_cs)
            cont = clean.copy()
            idx = rr.choice(n_cal_cs, n_cont, replace=False)
            cont[idx] = M_val
            test = gen_scores(dist, n_test_cs, rr)

            for mn in methods_cs:
                if 'MOM' in mn:
                    K = int(mn.split('(')[1].split(')')[0])
                    t = mom_fast(cont, K, tau, rr)
                elif 'Trim' in mn:
                    t = trimmed_t(cont, 1 - tau, eps_fixed)
                else:
                    t = split_t(cont, 1 - tau)
                cv_M[mn][rep] = cov(t, test)
                wd_M[mn][rep] = t

        cs_results[dist][M_val] = {
            m: {'cov': np.mean(cv_M[m]), 'cov_se': np.std(cv_M[m]) / np.sqrt(N_rep_cs),
                'width': np.mean(wd_M[m]), 'width_se': np.std(wd_M[m]) / np.sqrt(N_rep_cs)}
            for m in methods_cs}

    print(f"  {dist} done")

np.savez(os.path.join(OUT_DIR, 'contam_strength.npz'),
         M_vals_gauss=M_vals_gauss, M_vals_t3=M_vals_t3,
         cs_results=cs_results)
print("  -> results/contam_strength.npz saved.\n")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Conditional coverage analysis
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXPERIMENT 5: Conditional coverage")
print("=" * 70)

dist_cc = 'gauss'
n_cal_cc = 2000
n_test_cc = 2000  # larger test set for binning
eps_cc = [0, 0.05, 0.10]
N_rep_cc = 200
n_bins = 5
tau = 0.90

methods_cc = ['Split', 'MOM(10)', 'MOM(20)', 'MOM(40)', 'Trim(oracle)']
cc_results = {eps: {m: {'cov_bin': np.zeros((n_bins, N_rep_cc)),
                          'width_bin': np.zeros((n_bins, N_rep_cc))}
                     for m in methods_cc}
              for eps in eps_cc}

for ei, eps in enumerate(eps_cc):
    for rep in range(N_rep_cc):
        rr = np.random.default_rng(SEED * 80000 + ei * 500 + rep)
        clean = gen_scores(dist_cc, n_cal_cc, rr)
        cont = contam_high(clean, eps, dist_cc, rr)
        test = gen_scores(dist_cc, n_test_cc, rr)

        # Sort test scores into bins by quantile
        test_sorted = np.sort(test)
        bin_edges = np.linspace(0, n_test_cc, n_bins + 1, dtype=int)

        for mn in methods_cc:
            if 'MOM' in mn:
                K = int(mn.split('(')[1].split(')')[0])
                t = mom_fast(cont, K, tau, rr)
            elif 'Trim' in mn:
                t = trimmed_t(cont, 1 - tau, eps)
            else:
                t = split_t(cont, 1 - tau)

            for bi in range(n_bins):
                bin_test = test_sorted[bin_edges[bi]:bin_edges[bi + 1]]
                cc_results[eps][mn]['cov_bin'][bi, rep] = cov(t, bin_test)
                cc_results[eps][mn]['width_bin'][bi, rep] = t

    print(f"  eps={eps} done")

np.savez(os.path.join(OUT_DIR, 'conditional_coverage.npz'),
         eps_cc=eps_cc, n_bins=n_bins, cc_results=cc_results)
print("  -> results/conditional_coverage.npz saved.\n")

# ══════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Partition strategy ablation
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXPERIMENT 6: Partition strategy ablation")
print("=" * 70)

dist_ps = 'gauss'
n_cal_ps = 2000
n_test_ps = 500
eps_ps = [0, 0.05, 0.10]
N_rep_ps = 200
K_ps = [10, 20, 40]
tau = 0.90

def partition_random(scores, K, rng):
    n = len(scores)
    bs = n // K
    perm = rng.permutation(n)
    return [scores[perm[i*bs:(i+1)*bs]] for i in range(K)]

def partition_stratified(scores, K, rng):
    """Stratified: sort, then interleave (every K-th goes to same block)."""
    n = len(scores)
    bs = n // K
    order = np.argsort(scores)
    blocks = []
    for k in range(K):
        idx = order[k * bs:(k + 1) * bs] if k < K - 1 else order[k * bs:]
        blocks.append(scores[idx])
    return blocks

def partition_contiguous(scores, K, rng):
    """Contiguous: sort, then consecutive blocks."""
    n = len(scores)
    bs = n // K
    order = np.argsort(scores)
    blocks = []
    for k in range(K):
        idx = order[k * bs:(k + 1) * bs] if k < K - 1 else order[k * bs:]
        blocks.append(scores[idx])
    return blocks

partition_strategies = {
    'Random': partition_random,
    'Stratified': partition_stratified,
    'Contiguous': partition_contiguous,
}

ps_results = {}
for pname, pfn in partition_strategies.items():
    ps_results[pname] = {}
    for Ki, K in enumerate(K_ps):
        ps_results[pname][K] = {eps: {'cov': np.zeros(N_rep_ps), 'width': np.zeros(N_rep_ps)}
                                 for eps in eps_ps}
        for ei, eps in enumerate(eps_ps):
            for rep in range(N_rep_ps):
                rr = np.random.default_rng(SEED * 90000 + Ki * 500 + ei * 200 + rep)
                clean = gen_scores(dist_ps, n_cal_ps, rr)
                cont = contam_high(clean, eps, dist_ps, rr)
                test = gen_scores(dist_ps, n_test_ps, rr)

                # Partition and compute block quantiles
                blocks = pfn(cont, K, rr)
                block_q = np.array([eq(b, tau) for b in blocks])
                t = np.median(block_q)

                ps_results[pname][K][eps]['cov'][rep] = cov(t, test)
                ps_results[pname][K][eps]['width'][rep] = t

    print(f"  {pname} done")

np.savez(os.path.join(OUT_DIR, 'partition_strategy.npz'),
         K_ps=K_ps, eps_ps=eps_ps, ps_results=ps_results)
print("  -> results/partition_strategy.npz saved.\n")

print("=" * 70)
print("ALL NEW EXPERIMENTS COMPLETE")
print(f"Results saved to: {OUT_DIR}")
print("=" * 70)
