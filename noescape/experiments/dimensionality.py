"""
Effective Dimensionality Analysis Module

Computes d_eff via participation ratio and Levina-Bickel estimator.
Generates eigenvalue spectra and saves per-architecture results.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional
from noescape.utils import compute_participation_ratio, levina_bickel_estimator


def run_dimensionality(architecture, config: dict, items: Optional[List[str]] = None,
                        n_samples: int = 10000) -> Dict:
    """
    Compute effective dimensionality for an architecture.

    Returns d_eff (participation ratio), d_eff_lb (Levina-Bickel),
    d_95, d_99, eigenvalue spectrum.
    """
    if items is None:
        from noescape.utils import load_wikipedia_sentences
        wiki = load_wikipedia_sentences(n_sentences=n_samples, n_articles=500)
        items = [s['text'] for s in wiki[:n_samples]]
    else:
        items = items[:n_samples]

    embeddings = architecture.encode(items)

    pr_result = compute_participation_ratio(embeddings)
    lb_deff = levina_bickel_estimator(embeddings[:min(5000, len(embeddings))])

    return {
        'architecture': architecture.arch_key,
        'd_nominal': pr_result['d_nominal'],
        'd_eff': pr_result['d_eff'],
        'd_eff_lb': float(lb_deff),
        'd_95': pr_result['d_95'],
        'd_99': pr_result['d_99'],
        'eigenvalues_top100': pr_result['eigenvalues_top100'],
        'explained_variance_ratio': pr_result['explained_variance_ratio'],
        'n_samples': len(items),
    }


def save_results(result: dict, output_dir: str = 'results/dimensionality'):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{result['architecture']}.json")
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved dimensionality results to {path}")
