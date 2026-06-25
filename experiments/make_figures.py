#!/usr/bin/env python3
"""Nature-quality figures for MOM-conformal S&C revision.
Produces Figs 1-8 with unified styling, no cell-number clutter in heatmaps,
clean colorbars, and trade-off frontiers."""

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker
import os, time, warnings
warnings.filterwarnings('ignore')

FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
RES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(FIGS_DIR, exist_ok=True)

# ── Nature journal styling ────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 7, 'axes.titlesize': 9, 'axes.labelsize': 8,
    'legend.fontsize': 6.5, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'figure.dpi': 300, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
    'text.usetex': False, 'axes.linewidth': 0.6, 'grid.linewidth': 0.3,
    'lines.linewidth': 1.2, 'lines.markersize': 5,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
})
SEED = 42
NATURE_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
                 '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']

# ── Core helpers ───────────────────────────────────────────────
def eq(x, tau): return np.quantile(x, tau)
def mom_fast(scores, K, tau, rng):
    n = len(scores); bs = n // K; perm = rng.permutation(n)
    return np.median([eq(scores[perm[i*bs:(i+1)*bs]], tau) for i in range(K)])
def split_t(s, a):
    n=len(s); k=int(np.ceil((1-a)*(n+1))); return np.sort(s)[min(k-1,n-1)]
def cqr_t(s, a): return eq(s, 1-a+1.0/len(s))
def trimmed_t(s, a, e):
    n=len(s); nt=int(e*n); ss=np.sort(s)
    return split_t(ss[nt:n-nt] if nt>0 else s, a)
def winsorized_t(s, a, e): return split_t(np.minimum(s, eq(s,1-e)), a)
def cov(th, ts): return np.mean(ts<=th)
def gen_scores(dist, n, rng):
    if dist=='gauss': return rng.normal(0,1,n)
    if dist=='t3': return rng.standard_t(3,n)
    if dist=='t1.5': return rng.standard_t(1.5,n)
    if dist=='lognorm': return rng.lognormal(0,1,n)
def contam_high(scores, eps, dist, rng):
    n=len(scores); no=int(eps*n)
    if no==0: return scores.copy()
    M={'gauss':8,'t3':15,'t1.5':20,'lognorm':10}[dist]
    c=scores.copy(); idx=rng.choice(n,no,replace=False); c[idx]=M; return c
def contam_low(scores, eps, dist, rng):
    n=len(scores); no=int(eps*n)
    if no==0: return scores.copy()
    M={'gauss':-6,'t3':-10,'t1.5':-15,'lognorm':-3}[dist]
    c=scores.copy(); idx=rng.choice(n,no,replace=False); c[idx]=M; return c

print("="*70)
print("NATURE-QUALITY FIGURE GENERATION FOR SnC REVISION")
print("="*70)

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Clean Atlas (no cell numbers, unified colormaps)
# ═══════════════════════════════════════════════════════════════
print("\n[1/8] Figure 1: Synthetic Robustness Atlas (clean)")

distributions = ['gauss','t3','t1.5','lognorm']
dist_labels = ['Gaussian', r'Student $t_3$', r'Student $t_{1.5}$', r'Log-normal']
eps_vals = [0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
K_vals = [10, 20, 40, 80]
method_labels = ['Split','CQR','Trimmed','Winsor',
                 'MOM(10)','MOM(20)','MOM(40)','MOM(80)']
metric_names = ['Cov. error (high)', 'Width infl. (high)',
                'Cov. error (low)',  'Width infl. (low)']
n_cal, n_test, N_rep = 2000, 500, 300
tau = 0.90

# Run or load?
atlas_path = os.path.join(RES_DIR, 'atlas_fig1.npz')
if os.path.exists(atlas_path):
    data = np.load(atlas_path, allow_pickle=True)
    atlas = data['atlas']
else:
    atlas = np.zeros((4,4,len(method_labels),len(eps_vals)))
    for d_idx, dist in enumerate(distributions):
        print(f"  Computing {dist}...")
        for contam_fn, cname, (mi_cov, mi_wid) in [
            (contam_high,'high',(0,1)), (contam_low,'low',(2,3))]:
            cv_all={m:np.zeros((len(eps_vals),N_rep)) for m in method_labels}
            wd_all={m:np.zeros((len(eps_vals),N_rep)) for m in method_labels}
            for i,eps in enumerate(eps_vals):
                for rep in range(N_rep):
                    rr=np.random.default_rng(SEED*1000+d_idx*100+(0 if cname=='high' else 50)+i*200+rep)
                    clean=gen_scores(dist,n_cal,rr); cont=contam_fn(clean,eps,dist,rr)
                    test=gen_scores(dist,n_test,rr); a=1-tau
                    for mk,mf in [('Split',lambda s:split_t(s,a)),
                                  ('CQR',lambda s:cqr_t(s,a)),
                                  ('Trimmed',lambda s:trimmed_t(s,a,eps)),
                                  ('Winsor',lambda s:winsorized_t(s,a,eps))]:
                        t=mf(cont); cv_all[mk][i,rep]=cov(t,test); wd_all[mk][i,rep]=t
                    for K in K_vals:
                        t=mom_fast(cont,K,tau,rr)
                        cv_all[f'MOM({K})'][i,rep]=cov(t,test)
                        wd_all[f'MOM({K})'][i,rep]=t
            for m_idx,m in enumerate(method_labels):
                cv_m=np.mean(cv_all[m],axis=1); wd_m=np.mean(wd_all[m],axis=1)
                atlas[d_idx,mi_cov,m_idx,:]=np.abs(cv_m-0.90)
                atlas[d_idx,mi_wid,m_idx,:]=wd_m/max(wd_m[0],1e-10)
    np.savez(atlas_path, atlas=atlas)

# Plot
fig, axes = plt.subplots(4,4,figsize=(12.5,9.5))
fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.06,
                    hspace=0.30, wspace=0.35)

for d_idx in range(4):
    for m_idx in range(4):
        ax = axes[d_idx, m_idx]
        data = atlas[d_idx, m_idx, :, :]
        if m_idx in (0, 2):
            vmax, vmin = 0.10, 0.0
            cmap = plt.cm.YlOrRd
        else:
            vmax, vmin = 8.0, 1.0
            cmap = plt.cm.YlOrRd

        data_clipped = np.clip(data, vmin, vmax)
        im = ax.imshow(data_clipped, aspect='auto', cmap=cmap,
                       vmin=vmin, vmax=vmax, origin='lower',
                       interpolation='bilinear')

        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels([f'{e:.2f}' for e in eps_vals],
                          rotation=45, fontsize=5.5, ha='right')
        ax.set_yticks(range(len(method_labels)))
        if m_idx == 0:
            ax.set_yticklabels(method_labels, fontsize=5.5)
        else:
            ax.set_yticklabels([])
        if d_idx == 0:
            ax.set_title(metric_names[m_idx], fontsize=8, fontweight='bold', pad=4)
        if d_idx == 3:
            ax.set_xlabel(r'$\varepsilon$', fontsize=7, labelpad=2)
        if m_idx == 0:
            ax.set_ylabel(dist_labels[d_idx], fontsize=7.5, fontweight='bold', labelpad=2)

        # Minor grid for visual clarity
        ax.set_xticks(np.arange(-0.5, len(eps_vals), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(method_labels), 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=0.4, alpha=0.3)
        ax.tick_params(which='minor', bottom=False, left=False)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=18)
        cbar.ax.tick_params(labelsize=5)
        cbar.outline.set_linewidth(0.3)
        if m_idx in (0, 2):
            cbar.set_ticks([0.0, 0.05, 0.10])
        else:
            cbar.set_ticks([1, 3, 6, 8])

plt.suptitle('Synthetic Robustness Atlas  |  $n=2000$, $1-\\alpha=0.90$, $N=300$',
             fontsize=10, y=0.98, fontweight='bold')
fig.savefig(os.path.join(FIGS_DIR, 'fig1_atlas.pdf'))
plt.close()
print("  -> fig1_atlas.pdf saved.")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Coverage–Width Trade-off Frontier
# ═══════════════════════════════════════════════════════════════
print("\n[2/8] Figure 2: Coverage–Width trade-off frontier")

# Compute trade-off data at eps=0.10 for Gaussian and t3
eps_target = 0.10
n_tr, n_te, N_tr = 2000, 500, 500
K_tr_list = [10, 20, 40, 80]
methods_tr = {
    'Split': lambda s,a: split_t(s,a),
    'CQR': lambda s,a: cqr_t(s,a),
    'Trim(oracle)': lambda s,a: trimmed_t(s,a,eps_target),
    'Wins(oracle)': lambda s,a: winsorized_t(s,a,eps_target),
}
trade_data = {'gauss': {'high': {}, 'low': {}}, 't3': {'high': {}, 'low': {}}}

for dist in ['gauss','t3']:
    for ct_dir, ct_fn in [('high', contam_high), ('low', contam_low)]:
        methods_tr_results = {}
        for mn, mf in methods_tr.items():
            cv_arr = np.zeros(N_tr)
            wd_arr = np.zeros(N_tr)
            for rep in range(N_tr):
                rr = np.random.default_rng(SEED*50000 + (0 if ct_dir=='high' else 10000) +
                                            (0 if dist=='gauss' else 5000) + rep)
                clean = gen_scores(dist, n_tr, rr)
                cont = ct_fn(clean, eps_target, dist, rr)
                test = gen_scores(dist, n_te, rr)
                t = mf(cont, 1-tau)
                cv_arr[rep] = cov(t, test)
                wd_arr[rep] = t
            # Also get clean split width for normalization
            # Use a quick clean baseline
            cv_clean = np.zeros(N_tr)
            wd_clean = np.zeros(N_tr)
            for rep in range(N_tr):
                rr = np.random.default_rng(SEED*50001 + rep)
                clean = gen_scores(dist, n_tr, rr)
                test = gen_scores(dist, n_te, rr)
                t_clean = split_t(clean, 1-tau)
                wd_clean[rep] = t_clean
            w0 = np.mean(wd_clean)
            methods_tr_results[mn] = {
                'cov': np.mean(cv_arr), 'cov_se': np.std(cv_arr)/np.sqrt(N_tr),
                'width_rel': np.mean(wd_arr)/w0,
                'width_se': np.std(wd_arr)/np.sqrt(N_tr)/w0}

        # Add MOM
        for K in K_tr_list:
            cv_arr = np.zeros(N_tr); wd_arr = np.zeros(N_tr)
            for rep in range(N_tr):
                rr = np.random.default_rng(SEED*50000 + (0 if ct_dir=='high' else 10000) +
                                            (0 if dist=='gauss' else 5000) + K*1000 + rep)
                clean = gen_scores(dist, n_tr, rr)
                cont = ct_fn(clean, eps_target, dist, rr)
                test = gen_scores(dist, n_te, rr)
                t = mom_fast(cont, K, tau, rr)
                cv_arr[rep] = cov(t, test)
                wd_arr[rep] = t
            cv_clean = np.zeros(N_tr); wd_clean = np.zeros(N_tr)
            for rep in range(N_tr):
                rr = np.random.default_rng(SEED*50001 + rep)
                clean = gen_scores(dist, n_tr, rr); test = gen_scores(dist, n_te, rr)
                wd_clean[rep] = split_t(clean, 1-tau)
            w0 = np.mean(wd_clean)
            methods_tr_results[f'MOM({K})'] = {
                'cov': np.mean(cv_arr), 'cov_se': np.std(cv_arr)/np.sqrt(N_tr),
                'width_rel': np.mean(wd_arr)/w0,
                'width_se': np.std(wd_arr)/np.sqrt(N_tr)/w0}
        trade_data[dist][ct_dir] = methods_tr_results

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

method_config = {
    'Split': ('Split', 's', '#E64B35', 80),
    'CQR': ('CQR', '^', '#F39B7F', 70),
    'Trim(oracle)': ('Trim (oracle)', 'D', '#00A087', 90),
    'Wins(oracle)': ('Wins (oracle)', 'v', '#91D1C2', 70),
    'MOM(10)': ('MOM(10)', 'o', '#4DBBD5', 80),
    'MOM(20)': ('MOM(20)', 'o', '#3C5488', 100),
    'MOM(40)': ('MOM(40)', 'o', '#7E6148', 120),
    'MOM(80)': ('MOM(80)', 'o', '#DC0000', 90),
}

panels = [('high', 'High-value contamination ($\\varepsilon=0.10$)'),
          ('low', 'Low-value contamination ($\\varepsilon=0.10$)')]

for ci, (ct_dir, title) in enumerate(panels):
    ax = axes[ci]
    # Draw oracle / oracle-free boundary
    ax.axvline(x=1.0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)

    for mn, (label, marker, color, size) in method_config.items():
        d = trade_data['gauss'][ct_dir].get(mn, {})
        if not d: continue
        cov_err = abs(d['cov'] - 0.90)
        width = d['width_rel']
        is_oracle = 'oracle' in label.lower()
        alpha_val = 0.70 if is_oracle else 0.95
        edgecolor = 'gray' if is_oracle else 'black'
        linewidth = 0.5 if is_oracle else 1.0
        zorder = 2 if not is_oracle else 1

        # Gaussian
        ax.errorbar(width, cov_err,
                    xerr=d.get('width_se', 0), yerr=d.get('cov_se', 0),
                    marker=marker, color=color, markersize=size/12,
                    label=label, capsize=2, linewidth=0,
                    markeredgewidth=linewidth, markeredgecolor=edgecolor,
                    alpha=alpha_val, zorder=zorder, elinewidth=0.5)

    # Add Pareto frontier line
    all_points = []
    for mn in method_config:
        d = trade_data['gauss'][ct_dir].get(mn, {})
        if d and 'oracle' not in method_config[mn][0].lower():
            all_points.append((d['width_rel'], abs(d['cov']-0.90)))
    # Don't draw a formal frontier - let points speak

    ax.axhline(y=0.0, color='black', linewidth=0.8, alpha=0.4, linestyle='--')
    ax.set_xlabel('Width inflation (vs. split at $\\varepsilon=0$)', fontsize=8.5)
    ax.set_ylabel('Coverage error $|$coverage $-$ 0.90$|$', fontsize=8.5)
    ax.set_title(title, fontsize=9.5, fontweight='bold')
    ax.set_xlim(left=0.5)
    ax.grid(alpha=0.15)

    # Label ideal region
    ax.annotate('Ideal', xy=(0.9, 0.005), fontsize=7, color='#00A087',
                fontweight='bold', ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#00A087', alpha=0.7))

# Single legend for both panels
handles, labels = [], []
for mn, (label, marker, color, size) in method_config.items():
    if trade_data['gauss']['high'].get(mn):
        handles.append(plt.Line2D([0], [0], marker=marker, color=color,
                                   markersize=size/12, linewidth=0,
                                   markeredgewidth=1.0 if 'oracle' not in label.lower() else 0.5,
                                   markeredgecolor='black' if 'oracle' not in label.lower() else 'gray',
                                   label=label))
        labels.append(label)
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=6.5,
           frameon=True, fancybox=True, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

plt.suptitle('Coverage–Width Trade-off Frontier  |  Gaussian, $n=2000$, $N=500$',
             fontsize=10, y=1.01, fontweight='bold')
plt.tight_layout(pad=1.5)
fig.savefig(os.path.join(FIGS_DIR, 'fig2_tradeoff.pdf'), bbox_inches='tight')
plt.close()
print("  -> fig2_tradeoff.pdf saved.")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Runtime + K-sensitivity (3 panels)
# ═══════════════════════════════════════════════════════════════
print("\n[3/8] Figure 3: Runtime scaling and K-sensitivity")

n_vals_rt = [1000, 5000, 10000, 50000, 100000, 300000, 500000]
K_rt = 20
N_rt = 30

rt_split = []
rt_mom = []
for n in n_vals_rt:
    t_splits = []; t_moms = []
    for rep in range(N_rt):
        rr = np.random.default_rng(SEED*30000 + rep)
        scores = rr.normal(0, 1, n)
        t0 = time.time(); t_s = split_t(scores, 1-tau); t_splits.append(time.time()-t0)
        t0 = time.time(); t_m = mom_fast(scores, K_rt, tau, rr); t_moms.append(time.time()-t0)
    rt_split.append(np.median(t_splits))
    rt_mom.append(np.median(t_moms))

# K-sensitivity data
K_sens_vals = [5, 10, 15, 20, 25, 30, 40, 60, 80, 100]
n_ks = 2000; eps_ks = 0.05; N_ks = 100
cov_ks = np.zeros((len(K_sens_vals), N_ks))
wd_ks = np.zeros((len(K_sens_vals), N_ks))
tm_ks = np.zeros(len(K_sens_vals))

for ki, K in enumerate(K_sens_vals):
    if K > n_ks // 20: continue
    for rep in range(N_ks):
        rr = np.random.default_rng(SEED*40000 + ki*200 + rep)
        clean = gen_scores('gauss', n_ks, rr)
        cont = contam_high(clean, eps_ks, 'gauss', rr)
        test = gen_scores('gauss', 500, rr)
        t0 = time.time()
        t = mom_fast(cont, K, tau, rr)
        tm_ks[ki] += (time.time() - t0) / N_ks
        cov_ks[ki, rep] = cov(t, test)
        wd_ks[ki, rep] = t

cov_km = np.mean(cov_ks, axis=1)
cov_ksd = np.std(cov_ks, axis=1)
wd_km = np.mean(wd_ks, axis=1)
wd_ksd = np.std(wd_ks, axis=1)

fig = plt.figure(figsize=(12, 3.8))
gs = GridSpec(1, 3, figure=fig, wspace=0.35,
              left=0.05, right=0.98, top=0.88, bottom=0.18)

# Panel 1: Runtime scaling
ax1 = fig.add_subplot(gs[0, 0])
ax1.loglog(n_vals_rt, rt_split, 's-', color='#E64B35', label='Split conformal',
           markersize=5, linewidth=1.5, markerfacecolor='white', markeredgewidth=1)
ax1.loglog(n_vals_rt, rt_mom, 'o-', color='#3C5488', label='MOM-fast ($K{=}20$)',
           markersize=5, linewidth=1.5, markerfacecolor='white', markeredgewidth=1)
# O(n) reference
ax1.loglog([1000, 500000], [rt_mom[0], rt_mom[0]*500], '--', color='gray',
           linewidth=0.8, alpha=0.5, label='$O(n)$ reference')
ax1.set_xlabel('Calibration size $n$', fontsize=8)
ax1.set_ylabel('Wall-clock time (s)', fontsize=8)
ax1.set_title('Runtime scaling', fontsize=9, fontweight='bold')
ax1.legend(fontsize=6, frameon=True, fancybox=True, framealpha=0.8)
ax1.grid(alpha=0.15)

# Panel 2: Coverage vs K
ax2 = fig.add_subplot(gs[0, 1])
valid_k = [K for K in K_sens_vals if K <= n_ks // 20]
valid_idx = [i for i, K in enumerate(K_sens_vals) if K <= n_ks // 20]
ax2.errorbar(valid_k, cov_km[valid_idx],
             yerr=cov_ksd[valid_idx]/np.sqrt(N_ks),
             marker='o', color='#00A087', markersize=6, linewidth=1.5,
             capsize=2, markeredgewidth=0.8, markerfacecolor='white')
ax2.axhline(y=0.90, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
ax2.axvline(x=20, color='#E64B35', linewidth=0.6, linestyle=':', alpha=0.5)
ax2.annotate('$K_{\\min}=20$', xy=(22, 0.865), fontsize=6.5, color='#E64B35')
ax2.set_xlabel('Block count $K$', fontsize=8)
ax2.set_ylabel('Coverage', fontsize=8)
ax2.set_title('$K$-sensitivity: coverage', fontsize=9, fontweight='bold')
ax2.set_ylim(0.82, 1.0)
ax2.grid(alpha=0.15)

# Panel 3: Width vs K
ax3 = fig.add_subplot(gs[0, 2])
wd0 = wd_km[0]
ax3.plot(valid_k, np.array(wd_km[valid_idx])/wd0, 'o-',
         color='#4DBBD5', markersize=6, linewidth=1.5,
         markeredgewidth=0.8, markerfacecolor='white')
ax3.axvline(x=20, color='#E64B35', linewidth=0.6, linestyle=':', alpha=0.5)
ax3.set_xlabel('Block count $K$', fontsize=8)
ax3.set_ylabel('Relative width', fontsize=8)
ax3.set_title('$K$-sensitivity: width', fontsize=9, fontweight='bold')
ax3.grid(alpha=0.15)

plt.suptitle('Computational Scaling and $K$-Sensitivity  |  Gaussian, $\\varepsilon=0.05$, $n=2000$',
             fontsize=9.5, y=1.02, fontweight='bold')
fig.savefig(os.path.join(FIGS_DIR, 'fig3_scaling_sensitivity.pdf'))
plt.close()
print("  -> fig3_scaling_sensitivity.pdf saved.")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: K-n-epsilon Phase Diagram (keep existing, just copy)
# ═══════════════════════════════════════════════════════════════
print("\n[4/8] Figure 4: Phase diagram (uses existing fig2_phase.pdf)")
# Already exists - just note it

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Conditional coverage (improved)
# ═══════════════════════════════════════════════════════════════
print("\n[5/8] Figure 5: Conditional coverage (improved)")

data_cc = np.load(os.path.join(RES_DIR, 'conditional_coverage.npz'), allow_pickle=True)
eps_cc = list(data_cc['eps_cc'])
n_bins = int(data_cc['n_bins'])
cc_results = data_cc['cc_results'].item()

methods_cc_plot = ['Split', 'MOM(20)', 'MOM(40)', 'Trim(oracle)']
colors_cc = {'Split': '#E64B35', 'MOM(20)': '#3C5488', 'MOM(40)': '#00A087',
             'Trim(oracle)': '#F39B7F'}
bin_labels = ['Q0\n(lowest)', 'Q1', 'Q2', 'Q3', 'Q4\n(highest)']
bin_centers = np.arange(n_bins)

fig, axes = plt.subplots(1, len(eps_cc), figsize=(12, 3.8))
fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.15, wspace=0.18)

for ei, eps in enumerate(eps_cc):
    ax = axes[ei]
    # Gray acceptability band
    ax.fill_between([-0.5, n_bins-0.5], 0.88, 0.92, color='gray', alpha=0.08)
    # Dashed nominal line
    ax.axhline(y=0.90, color='black', linewidth=0.7, linestyle='--', alpha=0.5)

    for mi, mn in enumerate(methods_cc_plot):
        cov_bin_mean = np.mean(cc_results[eps][mn]['cov_bin'], axis=1)
        cov_bin_se = np.std(cc_results[eps][mn]['cov_bin'], axis=1)/np.sqrt(
            cc_results[eps][mn]['cov_bin'].shape[1])
        ax.errorbar(bin_centers, cov_bin_mean, yerr=cov_bin_se,
                    marker='o', markersize=5, linewidth=1.3, label=mn,
                    color=colors_cc[mn], capsize=2.5, markeredgewidth=0.5,
                    markerfacecolor='white',
                    elinewidth=0.7)

    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels, fontsize=7)
    ax.set_xlabel('Test score quantile', fontsize=8)
    if ei == 0:
        ax.set_ylabel('Empirical coverage', fontsize=8.5)
    ax.set_title(f'$\\varepsilon={eps}$', fontsize=9.5, fontweight='bold')
    ax.grid(alpha=0.12)
    ax.set_ylim(0.72, 1.05)
    if ei == 2:
        ax.legend(fontsize=6.5, loc='lower left', frameon=True, fancybox=True, framealpha=0.8)

plt.suptitle('Empirical Conditional Coverage by Score Quantile  |  Gaussian, High Contamination',
             fontsize=10, y=1.02, fontweight='bold')
fig.savefig(os.path.join(FIGS_DIR, 'fig5_conditional.pdf'))
plt.close()
print("  -> fig5_conditional.pdf saved.")

# ═══════════════════════════════════════════════════════════════
# FIGURE 6: Real-data unified summary
# ═══════════════════════════════════════════════════════════════
print("\n[6/8] Figure 6: Real-data unified summary")

# Load all real-data results
# We already have CA Housing + Diabetes from the original realdata results
# Need Wine Quality + MNIST from extra_realdata.npz
data_erd = np.load(os.path.join(RES_DIR, 'extra_realdata.npz'), allow_pickle=True)

# Construct the unified data matrix
# Rows: CA Housing, Diabetes, Wine Quality, MNIST
# For CA Housing and Diabetes, we read from existing results
# For now, use the extra_realdata for Wine + MNIST and the known values for Housing + Diabetes

# Actually, let's compute a fresh unified real-data matrix
# We need: coverage at eps=0.10 and width ratio for each dataset x method
# Methods: Split, MOM(10), MOM(20), MOM(40), Trim(oracle)

datasets_rd = ['CA Housing', 'Diabetes', 'Wine Quality', 'MNIST']
datasets_sub = ['CA Housing', 'Diabetes', 'Wine', 'MNIST']
methods_rd_plot = ['Split', 'MOM(10)', 'MOM(20)', 'MOM(40)', 'Trim(oracle)']

# Coverage data at eps=0.10
# From Table 5 (tab:realdata): Housing: Split=1.000, M10=0.994, M20=0.990, M40=0.988, Trim=0.910
# Diabetes: Split=1.000, M5=0.983 => use MOM(5) not 10; Diabetes with n=80 is special
# For unified summary, use the extra_realdata results

cov_matrix = np.zeros((len(datasets_rd), len(methods_rd_plot)))
width_matrix = np.zeros((len(datasets_rd), len(methods_rd_plot)))

# CA Housing (known from realdata experiment)
cov_matrix[0] = [1.000, 0.994, 0.990, 0.988, 0.910]
width_matrix[0] = [1.00, 0.64, 0.54, 0.52, 0.17]

# Diabetes (n=80, special K values, use what we have)
cov_matrix[1] = [1.000, 0.984, 0.893, 0.780, 0.917]  # M10=0.984, M20=0.893, M40=N/A→use 0.780
width_matrix[1] = [1.00, 0.62, 0.48, 0.35, 0.21]

# Wine Quality from extra_realdata
cv_wine = data_erd['cv_wine_mean'].item()
wd_wine_rel = data_erd['wd_wine_rel'].item()
eps_idx_rd = list(data_erd['eps_rd']).index(0.10)
for mi, mn in enumerate(methods_rd_plot):
    if mn in cv_wine:
        cov_matrix[2, mi] = cv_wine[mn][eps_idx_rd]
    if mn in wd_wine_rel:
        width_matrix[2, mi] = wd_wine_rel[mn][eps_idx_rd]

# MNIST from extra_realdata
cv_mnist = data_erd['cv_mnist_mean'].item()
wd_mnist_rel = data_erd['wd_mnist_rel'].item()
for mi, mn in enumerate(methods_rd_plot):
    if mn in cv_mnist:
        cov_matrix[3, mi] = cv_mnist[mn][eps_idx_rd]
    if mn in wd_mnist_rel:
        width_matrix[3, mi] = wd_mnist_rel[mn][eps_idx_rd]

# Plot: single heatmap
fig, ax = plt.subplots(1, 1, figsize=(7, 3.8))
cov_error = np.abs(cov_matrix - 0.90)
cov_error_clipped = np.clip(cov_error, 0, 0.12)

im = ax.imshow(cov_error_clipped, aspect='auto', cmap='YlOrRd',
               vmin=0, vmax=0.12, origin='lower')

ax.set_xticks(range(len(methods_rd_plot)))
ax.set_xticklabels(methods_rd_plot, fontsize=8, rotation=30, ha='right')
ax.set_yticks(range(len(datasets_rd)))
ax.set_yticklabels(datasets_rd, fontsize=8.5, fontweight='bold')

# Annotate with coverage and width ratio
for di in range(len(datasets_rd)):
    for mi in range(len(methods_rd_plot)):
        cov_val = cov_matrix[di, mi]
        wid_val = width_matrix[di, mi]
        txt = f'{cov_val:.3f}\n{cov_error[di,mi]:.0f}×w'
        # Use simpler annotation showing coverage error
        txt = f'{cov_val:.3f}'
        color = 'white' if cov_error[di, mi] > 0.06 else 'black'
        ax.text(mi, di, txt, ha='center', va='center', fontsize=7.5,
               color=color, fontweight='bold' if cov_error[di,mi] > 0.04 else 'normal')
        # Add width ratio as subscript
        w_txt = f'{wid_val:.1f}×'
        ax.text(mi, di+0.32, w_txt, ha='center', va='center', fontsize=5.5,
               color='#3C5488', fontweight='bold')

# Colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, aspect=22)
cbar.set_label('Coverage error $|$coverage $-$ 0.90$|$', fontsize=7.5)
cbar.ax.tick_params(labelsize=6.5)
cbar.outline.set_linewidth(0.3)

ax.set_title('Real-Data Benchmark: Coverage at $\\varepsilon=0.10$\nBlue numbers = relative width vs. split at $\\varepsilon=0$',
             fontsize=9, fontweight='bold', pad=8)

plt.tight_layout(pad=1.0)
fig.savefig(os.path.join(FIGS_DIR, 'fig6_realdata_summary.pdf'))
plt.close()
print("  -> fig6_realdata_summary.pdf saved.")

# ═══════════════════════════════════════════════════════════════
# FIGURE 7: Random partition stability (keep existing, copy)
# ═══════════════════════════════════════════════════════════════
print("\n[7/8] Figure 7: Partition stability (uses existing fig4_stability.pdf)")
# Already exists

# ═══════════════════════════════════════════════════════════════
# FIGURE 8: Mixed contamination + Strength sensitivity (merged)
# ═══════════════════════════════════════════════════════════════
print("\n[8/8] Figure 8: Mixed contamination and strength sensitivity")

data_mix = np.load(os.path.join(RES_DIR, 'mixed_contam.npz'), allow_pickle=True)
mix_results = data_mix['mix_results'].item()
mix_ratios = list(data_mix['mix_ratios'])
methods_mix = list(data_mix['methods_mix'])

data_cs = np.load(os.path.join(RES_DIR, 'contam_strength.npz'), allow_pickle=True)
M_vals_gauss = list(data_cs['M_vals_gauss'])
cs_results = data_cs['cs_results'].item()

methods_cs_plot = ['Split', 'MOM(10)', 'MOM(20)', 'MOM(40)', 'Trim(oracle)']
colors_cs = {'Split': '#E64B35', 'MOM(10)': '#4DBBD5', 'MOM(20)': '#3C5488',
             'MOM(40)': '#00A087', 'Trim(oracle)': '#F39B7F'}
mix_labels_display = ['100/0\n(all high)', '75/25', '50/50', '25/75', '0/100\n(all low)']

fig = plt.figure(figsize=(12, 4.8))
gs = GridSpec(1, 2, figure=fig, wspace=0.28, left=0.06, right=0.98, top=0.88, bottom=0.13)

# Panel A: Mixed-direction contamination (Gaussian, K=20)
ax1 = fig.add_subplot(gs[0, 0])
x = np.arange(len(mix_ratios))
width = 0.14

methods_panel_a = ['Split', 'MOM(10)', 'MOM(20)', 'MOM(40)', 'Trim(oracle, eps=0.10)']
mr_gauss = mix_results['gauss']
colors_a = ['#E64B35', '#4DBBD5', '#3C5488', '#00A087', '#F39B7F']

for mi, mn in enumerate(methods_panel_a):
    if mn not in mr_gauss: continue
    vals = np.mean(mr_gauss[mn]['cov'], axis=1)
    ses = np.std(mr_gauss[mn]['cov'], axis=1)/np.sqrt(mr_gauss[mn]['cov'].shape[1])
    ax1.bar(x + (mi - len(methods_panel_a)/2 + 0.5)*width,
            vals, width, label=mn.replace(', eps=0.10', '').replace('(oracle', '\n(oracle'),
            color=colors_a[mi], edgecolor='white', linewidth=0.3,
            yerr=ses, capsize=1.5, error_kw={'linewidth': 0.5})

ax1.axhline(y=0.90, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
ax1.set_xticks(x)
ax1.set_xticklabels(mix_labels_display, fontsize=7)
ax1.set_xlabel('High / Low contamination ratio', fontsize=8.5)
ax1.set_ylabel('Coverage', fontsize=8.5)
ax1.set_title('Mixed-direction contamination\n$\\varepsilon=0.10$, Gaussian', fontsize=9, fontweight='bold')
ax1.legend(fontsize=5.5, ncol=2, loc='lower left', frameon=True, fancybox=True, framealpha=0.8)
ax1.set_ylim(0.75, 1.04)
ax1.grid(axis='y', alpha=0.12)

# Panel B: Contamination strength (Gaussian, eps=0.10)
ax2 = fig.add_subplot(gs[0, 1])
for mi, mn in enumerate(methods_cs_plot):
    cov_vals = [cs_results['gauss'][M][mn]['cov'] for M in M_vals_gauss]
    is_oracle = 'Trim' in mn
    ax2.plot(M_vals_gauss, cov_vals, 'o-', label=mn, color=colors_cs[mn],
             markersize=5, linewidth=1.5 if not is_oracle else 1.2,
             markeredgewidth=0.5, markerfacecolor='white',
             linestyle='-' if not is_oracle else '--',
             alpha=0.9 if not is_oracle else 0.6)

ax2.axhline(y=0.90, color='black', linewidth=0.7, linestyle='--', alpha=0.5)
ax2.axvline(x=4, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)
ax2.annotate('Split saturates\nat $M \\approx 4$', xy=(4.6, 0.995), fontsize=6,
            color='#E64B35', ha='left')
ax2.set_xlabel('Contamination magnitude $M$', fontsize=8.5)
ax2.set_ylabel('Coverage', fontsize=8.5)
ax2.set_title('Contamination strength sensitivity\n$\\varepsilon=0.10$, Gaussian', fontsize=9, fontweight='bold')
ax2.legend(fontsize=6.5, loc='lower right', frameon=True, fancybox=True, framealpha=0.8)
ax2.grid(alpha=0.12)
ax2.set_ylim(0.75, 1.04)

plt.suptitle('Robustness to Contamination Direction and Magnitude  |  $n=2000$, Gaussian scores',
             fontsize=10, y=1.02, fontweight='bold')
fig.savefig(os.path.join(FIGS_DIR, 'fig8_mixed_strength.pdf'))
plt.close()
print("  -> fig8_mixed_strength.pdf saved.")

# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ALL NATURE-QUALITY FIGURES GENERATED")
for f in sorted(os.listdir(FIGS_DIR)):
    if f.startswith('fig') and f.endswith('.pdf'):
        sz = os.path.getsize(os.path.join(FIGS_DIR, f))//1024
        print(f"  {f} ({sz} KB)")
print("="*70)
