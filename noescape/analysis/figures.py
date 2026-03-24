"""
Figure generation for the No Escape paper.

All figures follow the Nature-quality style from the HIDE paper.
Generates PDF + PNG at 600 DPI.

Figure 1: No-Escape Overview (conceptual diagram)
Figure 2: Cross-Architecture Forgetting Curves (5 panels)
Figure 3: Universal Forgetting Exponent (b vs competitors)
Figure 4: Universal DRM False Recall
Figure 5: Effective Dimensionality Convergence
Figure 6: Solution Analysis (4 panels)
Figure 7: Architecture Comparison Heatmap
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


# ---- Style Setup (matching HIDE figure_style.py) ----

def set_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'lines.linewidth': 1.2,
        'lines.markersize': 4,
        'axes.linewidth': 0.6,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'axes.grid': False,
        'figure.dpi': 150,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'legend.frameon': False,
    })


COLORS = {
    'vector_db': '#2171B5',
    'attention': '#CB181D',
    'filesystem': '#238B45',
    'graph': '#6A51A3',
    'parametric': '#D94801',
    'human': '#525252',
}

ARCH_NAMES = {
    'vector_db': 'Vector DB',
    'attention': 'Attention',
    'filesystem': 'Filesystem',
    'graph': 'Graph',
    'parametric': 'Parametric',
}

BLUES_GRADIENT = ['#C6DBEF', '#9ECAE1', '#6BAED6', '#3182BD', '#08519C']
FULL_WIDTH = 180 / 25.4
HALF_WIDTH = 89 / 25.4


def panel_label(ax, label, x=-0.12, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif')


def human_reference_line(ax, value, label='Human', orientation='horizontal'):
    if orientation == 'horizontal':
        ax.axhline(y=value, color=COLORS['human'], linestyle='--',
                   linewidth=0.8, alpha=0.8, zorder=1)
        xlim = ax.get_xlim()
        ax.text(xlim[1], value, f'  {label}',
                color=COLORS['human'], fontsize=7, ha='left', va='center',
                fontstyle='italic', clip_on=False)


def save_figure(fig, name, figures_dir='figures'):
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, f'{name}.pdf'), format='pdf')
    fig.savefig(os.path.join(figures_dir, f'{name}.png'), format='png')
    plt.close(fig)
    print(f"Saved: {figures_dir}/{name}.pdf and .png")


# ---- Figure Generators ----

def figure2_forgetting_curves(results_dir: str, figures_dir: str = 'figures'):
    """Cross-Architecture Forgetting Curves — 5 panels."""
    set_nature_style()

    archs = ['vector_db', 'attention', 'filesystem', 'graph', 'parametric']
    fig, axes = plt.subplots(1, 5, figsize=(FULL_WIDTH, FULL_WIDTH / 5 * 1.1), sharey=True)

    labels = 'abcde'

    for i, arch in enumerate(archs):
        ax = axes[i]
        panel_label(ax, labels[i])
        ax.set_title(ARCH_NAMES[arch], fontsize=8)

        result_path = os.path.join(results_dir, arch, 'ebbinghaus.json')
        if not os.path.exists(result_path):
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        with open(result_path) as f:
            data = json.load(f)

        agg = data.get('aggregated', {}).get('per_n_near', {})
        n_near_keys = sorted(agg.keys(), key=lambda x: int(x))

        # Plot forgetting curves at different competitor counts
        n_colors = len(n_near_keys)
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_colors))

        for j, n_key in enumerate(n_near_keys):
            # Get one seed's data for the curve
            first_seed = list(data['per_seed'].keys())[0]
            seed_data = data['per_seed'][first_seed]['per_n_near'].get(n_key, {})
            ages = seed_data.get('ages', [])
            accs = seed_data.get('accuracies', [])
            if ages and accs:
                ax.plot(ages, accs, color=colors[j], linewidth=0.8,
                       label=f'n={n_key}' if j % 2 == 0 else None, alpha=0.8)

        # Human reference
        human_ages = np.linspace(0.1, 30, 50)
        human_curve = 0.9 * human_ages ** (-0.5) + 0.1
        ax.plot(human_ages, np.clip(human_curve, 0, 1), '--',
               color=COLORS['human'], linewidth=0.8, label='Human')

        ax.set_xlabel('Age (days)')
        if i == 0:
            ax.set_ylabel('Retrieval accuracy')
        ax.set_xlim(0, 31)
        ax.set_ylim(0, 1.05)

        if i == 0:
            ax.legend(loc='upper right', fontsize=5)

    fig.tight_layout()
    save_figure(fig, 'figure2_forgetting_curves', figures_dir)


def figure3_universal_b(results_dir: str, figures_dir: str = 'figures'):
    """Universal Forgetting Exponent — b vs competitor count for all architectures."""
    set_nature_style()

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.8))
    panel_label(ax, 'a')

    archs = ['vector_db', 'attention', 'filesystem', 'graph', 'parametric']

    for arch in archs:
        result_path = os.path.join(results_dir, arch, 'ebbinghaus.json')
        if not os.path.exists(result_path):
            continue

        with open(result_path) as f:
            data = json.load(f)

        agg = data.get('aggregated', {}).get('per_n_near', {})
        n_values = []
        b_means = []
        b_cis_lo = []
        b_cis_hi = []

        for n_key in sorted(agg.keys(), key=lambda x: int(x)):
            n_values.append(int(n_key))
            b_means.append(agg[n_key]['b_mean'])
            b_cis_lo.append(agg[n_key].get('b_ci_lower', agg[n_key]['b_mean'] - agg[n_key]['b_std']))
            b_cis_hi.append(agg[n_key].get('b_ci_upper', agg[n_key]['b_mean'] + agg[n_key]['b_std']))

        n_values = np.array(n_values)
        b_means = np.array(b_means)
        b_lo = np.array(b_cis_lo)
        b_hi = np.array(b_cis_hi)

        ax.plot(n_values, b_means, 'o-', color=COLORS[arch], markersize=3,
               label=ARCH_NAMES[arch], linewidth=1.0)
        ax.fill_between(n_values, b_lo, b_hi, color=COLORS[arch], alpha=0.15)

    # Human reference
    human_reference_line(ax, 0.5, 'Human b ≈ 0.5')

    ax.set_xlabel('Number of competitors')
    ax.set_ylabel('Forgetting exponent b')
    ax.set_xscale('log')
    ax.legend(loc='lower right', fontsize=6)
    ax.set_ylim(-0.05, 1.0)

    fig.tight_layout()
    save_figure(fig, 'figure3_universal_b', figures_dir)


def figure4_drm(results_dir: str, figures_dir: str = 'figures'):
    """Universal DRM False Recall — bar chart + threshold curves."""
    set_nature_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.35))
    panel_label(ax1, 'a')
    panel_label(ax2, 'b')

    archs = ['vector_db', 'attention', 'filesystem', 'graph', 'parametric']
    x = np.arange(len(archs) + 1)  # +1 for human
    width = 0.25

    hits = []
    lure_fas = []
    unrel_fas = []

    for arch in archs:
        result_path = os.path.join(results_dir, arch, 'drm.json')
        if os.path.exists(result_path):
            with open(result_path) as f:
                data = json.load(f)
            agg = data.get('aggregated', {})
            hits.append(agg.get('hit_rate_mean', 0))
            lure_fas.append(agg.get('lure_fa_mean', 0))
            unrel_fas.append(agg.get('unrelated_fa_mean', 0))
        else:
            hits.append(0)
            lure_fas.append(0)
            unrel_fas.append(0)

    # Human values
    hits.append(0.86)
    lure_fas.append(0.55)
    unrel_fas.append(0.02)

    labels = [ARCH_NAMES[a] for a in archs] + ['Human']
    colors_list = [COLORS[a] for a in archs] + [COLORS['human']]

    ax1.bar(x - width, hits, width, label='Hit rate', color='#6BAED6', alpha=0.8)
    ax1.bar(x, lure_fas, width, label='Lure FA', color='#CB181D', alpha=0.8)
    ax1.bar(x + width, unrel_fas, width, label='Unrelated FA', color='#969696', alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
    ax1.set_ylabel('Rate')
    ax1.legend(fontsize=6)
    ax1.set_ylim(0, 1.1)

    # Panel b: lure FA comparison
    arch_lure_fas = lure_fas[:-1]
    for i, (arch, fa) in enumerate(zip(archs, arch_lure_fas)):
        ax2.barh(i, fa, color=COLORS[arch], alpha=0.8)
    ax2.barh(len(archs), 0.55, color=COLORS['human'], alpha=0.8)

    ax2.set_yticks(range(len(archs) + 1))
    ax2.set_yticklabels([ARCH_NAMES[a] for a in archs] + ['Human'], fontsize=6)
    ax2.set_xlabel('Lure false alarm rate')
    ax2.axvline(x=0.55, color=COLORS['human'], linestyle='--', linewidth=0.8, alpha=0.6)

    fig.tight_layout()
    save_figure(fig, 'figure4_drm', figures_dir)


def figure5_dimensionality(results_dir: str, figures_dir: str = 'figures'):
    """Effective Dimensionality Convergence."""
    set_nature_style()

    fig, ax = plt.subplots(figsize=(HALF_WIDTH, HALF_WIDTH * 0.8))
    panel_label(ax, 'a')

    archs = ['vector_db', 'attention', 'filesystem', 'graph', 'parametric']
    dim_dir = os.path.join(results_dir, 'dimensionality')

    for arch in archs:
        path = os.path.join(dim_dir, f'{arch}.json')
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            d_nom = data.get('d_nominal', 0)
            d_eff = data.get('d_eff', 0)
            ax.scatter(d_nom, d_eff, color=COLORS[arch], s=60, zorder=5,
                      label=f"{ARCH_NAMES[arch]} (d_eff={d_eff:.0f})")

    # Biological range
    ax.axhspan(10, 100, alpha=0.1, color='gray', label='Biological range')

    # d_eff = d_nom line
    ax.plot([0, 5000], [0, 5000], '--', color='gray', linewidth=0.5, alpha=0.5)

    # Interference threshold
    ax.axhline(y=100, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.text(4000, 110, 'Interference\nthreshold', fontsize=6, color='red', ha='center')

    ax.set_xlabel('Nominal dimensionality')
    ax.set_ylabel('Effective dimensionality')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=5, loc='upper left')

    fig.tight_layout()
    save_figure(fig, 'figure5_dimensionality', figures_dir)


def figure6_solutions(results_dir: str, figures_dir: str = 'figures'):
    """Solution Analysis — 4 panels showing tradeoff frontiers."""
    set_nature_style()

    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH, FULL_WIDTH * 0.7))
    labels = 'abcd'

    sol_path = os.path.join(results_dir, 'solutions', 'solution_analysis.json')
    if not os.path.exists(sol_path):
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        save_figure(fig, 'figure6_solutions', figures_dir)
        return

    with open(sol_path) as f:
        sol_data = json.load(f)

    # Panel a: High dimensionality
    ax = axes[0, 0]
    panel_label(ax, 'a')
    ax.set_title('High nominal dim', fontsize=8)
    hd = sol_data.get('high_dim', {})
    d_noms = []
    d_effs = []
    bs = []
    for d_key, d_val in sorted(hd.items(), key=lambda x: int(x[0])):
        d_noms.append(d_val['d_nominal'])
        d_effs.append(d_val['d_eff'])
        bs.append(d_val['fitted_b'])
    if d_noms:
        ax.plot(d_noms, bs, 'o-', color='#2171B5', markersize=4)
        ax.set_xlabel('Nominal dim')
        ax.set_ylabel('Forgetting exponent b')
        ax.set_xscale('log')

    # Panel b: BM25
    ax = axes[0, 1]
    panel_label(ax, 'b')
    ax.set_title('BM25 keyword retrieval', fontsize=8)
    bm = sol_data.get('bm25', {})
    if bm:
        tradeoff = bm.get('tradeoff', {})
        ax.bar(['Immunity', 'Usefulness'],
               [tradeoff.get('immunity', 0), tradeoff.get('usefulness', 0)],
               color=['#238B45', '#CB181D'], alpha=0.7)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score')

    # Panel c: Orthogonalization
    ax = axes[1, 0]
    panel_label(ax, 'c')
    ax.set_title('Orthogonalization', fontsize=8)
    orth = sol_data.get('orthogonalization', {})
    if orth:
        methods = []
        immunities = []
        usefulnesses = []
        for key, val in orth.items():
            if isinstance(val, dict):
                methods.append(key[:10])
                immunities.append(val.get('immunity', 0))
                usefulnesses.append(val.get('usefulness', val.get('semantic_accuracy', 0)))
        if methods:
            x = np.arange(len(methods))
            ax.bar(x - 0.15, immunities, 0.3, label='Immunity', color='#238B45', alpha=0.7)
            ax.bar(x + 0.15, usefulnesses, 0.3, label='Usefulness', color='#CB181D', alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=5)
            ax.legend(fontsize=5)
            ax.set_ylim(0, 1.1)

    # Panel d: Compression
    ax = axes[1, 1]
    panel_label(ax, 'd')
    ax.set_title('Memory compression', fontsize=8)
    comp = sol_data.get('compression', {})
    if comp:
        n_clusters = []
        bs_comp = []
        accuracies = []
        for key, val in sorted(comp.items(), key=lambda x: int(x[0])):
            n_clusters.append(val['n_clusters'])
            bs_comp.append(val['fitted_b'])
            accuracies.append(val['cluster_retrieval_accuracy'])
        if n_clusters:
            ax.plot(n_clusters, bs_comp, 'o-', color='#2171B5', label='b', markersize=3)
            ax2 = ax.twinx()
            ax2.plot(n_clusters, accuracies, 's-', color='#CB181D', label='Accuracy', markersize=3)
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('Forgetting exponent b', color='#2171B5')
            ax2.set_ylabel('Retrieval accuracy', color='#CB181D')
            ax.set_xscale('log')

    fig.tight_layout()
    save_figure(fig, 'figure6_solutions', figures_dir)


def figure7_heatmap(results_dir: str, figures_dir: str = 'figures'):
    """Architecture Comparison Heatmap."""
    set_nature_style()

    archs = ['vector_db', 'attention', 'filesystem', 'graph', 'parametric']
    phenomena = ['Forgetting b', 'DRM FA', 'Spacing d', 'TOT rate']
    human_values = [0.5, 0.55, 1.0, 0.036]  # Human reference values

    data_matrix = np.zeros((len(archs) + 1, len(phenomena)))
    data_matrix[-1] = human_values  # Human row

    for i, arch in enumerate(archs):
        # Forgetting b
        ebb_path = os.path.join(results_dir, arch, 'ebbinghaus.json')
        if os.path.exists(ebb_path):
            with open(ebb_path) as f:
                ebb = json.load(f)
            agg = ebb.get('aggregated', {}).get('per_n_near', {})
            if agg:
                max_key = max(agg.keys(), key=lambda x: int(x))
                data_matrix[i, 0] = agg[max_key].get('b_mean', 0)

        # DRM FA
        drm_path = os.path.join(results_dir, arch, 'drm.json')
        if os.path.exists(drm_path):
            with open(drm_path) as f:
                drm_data = json.load(f)
            data_matrix[i, 1] = drm_data.get('aggregated', {}).get('lure_fa_mean', 0)

        # Spacing Cohen's d
        sp_path = os.path.join(results_dir, arch, 'spacing.json')
        if os.path.exists(sp_path):
            with open(sp_path) as f:
                sp_data = json.load(f)
            data_matrix[i, 2] = sp_data.get('aggregated', {}).get('cohens_d_long_vs_massed', 0)

        # TOT rate
        tot_path = os.path.join(results_dir, arch, 'tot.json')
        if os.path.exists(tot_path):
            with open(tot_path) as f:
                tot_data = json.load(f)
            data_matrix[i, 3] = tot_data.get('aggregated', {}).get('tot_rate_mean', 0)

    fig, ax = plt.subplots(figsize=(HALF_WIDTH * 1.2, HALF_WIDTH * 0.9))
    panel_label(ax, 'a')

    # Normalize each column to human baseline
    normalized = np.zeros_like(data_matrix)
    for j in range(len(phenomena)):
        human_val = human_values[j]
        if human_val > 0:
            normalized[:, j] = data_matrix[:, j] / human_val
        else:
            normalized[:, j] = data_matrix[:, j]

    im = ax.imshow(normalized, cmap='RdYlGn', vmin=0, vmax=2, aspect='auto')

    row_labels = [ARCH_NAMES[a] for a in archs] + ['Human']
    ax.set_xticks(range(len(phenomena)))
    ax.set_xticklabels(phenomena, fontsize=7)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(phenomena)):
            val = data_matrix[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6)

    plt.colorbar(im, ax=ax, shrink=0.7, label='Relative to human')

    fig.tight_layout()
    save_figure(fig, 'figure7_heatmap', figures_dir)


def generate_all_figures(results_dir: str = 'results', figures_dir: str = 'figures'):
    """Generate all figures for the paper."""
    os.makedirs(figures_dir, exist_ok=True)

    print("Generating Figure 2: Cross-Architecture Forgetting Curves...")
    figure2_forgetting_curves(results_dir, figures_dir)

    print("Generating Figure 3: Universal Forgetting Exponent...")
    figure3_universal_b(results_dir, figures_dir)

    print("Generating Figure 4: Universal DRM False Recall...")
    figure4_drm(results_dir, figures_dir)

    print("Generating Figure 5: Effective Dimensionality Convergence...")
    figure5_dimensionality(results_dir, figures_dir)

    print("Generating Figure 6: Solution Analysis...")
    figure6_solutions(results_dir, figures_dir)

    print("Generating Figure 7: Architecture Comparison Heatmap...")
    figure7_heatmap(results_dir, figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == '__main__':
    generate_all_figures()
