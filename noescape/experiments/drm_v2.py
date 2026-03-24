"""
DRM False Memory Experiment v2 — Matches HIDE protocol exactly.

Key differences from v1:
- Threshold sweep from 0.50 to 0.95 step 0.01 (HIDE protocol)
- Reports FA at theta=0.82 AND at best-match theta
- Full threshold operating curve saved
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional

from noescape.utils import set_seed, bootstrap_confidence_interval, load_drm_word_lists


def run_drm_v2(architecture, config: dict, seed: int, drm_lists: dict = None) -> dict:
    """DRM experiment matching HIDE Phase 5 protocol."""
    set_seed(seed)

    if drm_lists is None:
        drm_lists = load_drm_word_lists(config)

    drm_cfg = config['experiments']['drm']
    theta_start = drm_cfg.get('theta_range', [0.5, 1.0])[0]
    theta_stop = drm_cfg.get('theta_range', [0.5, 1.0])[1]
    theta_steps = drm_cfg.get('theta_steps', 100)

    per_list = {}
    all_studied_sims = []
    all_lure_sims = []
    all_unrelated_sims = []

    list_names = list(drm_lists.keys())

    for list_name in list_names:
        data = drm_lists[list_name]
        studied = data['studied']
        lure = data['lure']

        # Get unrelated words from different lists
        other_lists = [k for k in list_names if k != list_name]
        unrelated = drm_lists[other_lists[0]]['studied'][:5]

        # Encode all words
        all_words = studied + [lure] + unrelated
        embeddings = architecture.encode(all_words)

        studied_embs = embeddings[:len(studied)]
        lure_emb = embeddings[len(studied)]
        unrelated_embs = embeddings[len(studied) + 1:]

        # Centroid of studied words (HIDE protocol)
        centroid = studied_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        # Cosine similarities to centroid
        studied_sims = [float(np.dot(e / (np.linalg.norm(e) + 1e-8), centroid)) for e in studied_embs]
        lure_sim = float(np.dot(lure_emb / (np.linalg.norm(lure_emb) + 1e-8), centroid))
        unrel_sims = [float(np.dot(e / (np.linalg.norm(e) + 1e-8), centroid)) for e in unrelated_embs]

        all_studied_sims.extend(studied_sims)
        all_lure_sims.append(lure_sim)
        all_unrelated_sims.extend(unrel_sims)

        per_list[list_name] = {
            'studied_mean_sim': float(np.mean(studied_sims)),
            'lure_sim': lure_sim,
            'unrelated_mean_sim': float(np.mean(unrel_sims)),
        }

    # Threshold sweep (HIDE: 0.50 to 0.95 step 0.01)
    thresholds = np.linspace(theta_start, theta_stop, theta_steps)
    operating_curve = []

    for theta in thresholds:
        hit = np.mean([s > theta for s in all_studied_sims])
        lure_fa = np.mean([s > theta for s in all_lure_sims])
        unrel_fa = np.mean([s > theta for s in all_unrelated_sims])
        operating_curve.append({
            'theta': float(theta),
            'hit_rate': float(hit),
            'lure_fa_rate': float(lure_fa),
            'unrelated_fa_rate': float(unrel_fa),
        })

    # Results at HIDE's canonical theta=0.82
    theta_082 = 0.82
    hit_082 = np.mean([s > theta_082 for s in all_studied_sims])
    lure_fa_082 = np.mean([s > theta_082 for s in all_lure_sims])
    unrel_fa_082 = np.mean([s > theta_082 for s in all_unrelated_sims])

    # Best match to human FA rate (HIDE: human_false_alarm_rate=0.55)
    human_fa = config.get('calibration', {}).get('drm_fa', {}).get('hide_value', 0.583)
    best_match_theta = theta_082
    best_match_diff = abs(lure_fa_082 - human_fa)
    for oc in operating_curve:
        diff = abs(oc['lure_fa_rate'] - human_fa)
        if diff < best_match_diff and oc['unrelated_fa_rate'] < 0.1:
            best_match_diff = diff
            best_match_theta = oc['theta']

    # Per-list results at theta=0.82
    for list_name in list_names:
        ls = per_list[list_name]['lure_sim']
        per_list[list_name]['lure_above_082'] = ls > theta_082
        per_list[list_name]['lure_above_best'] = ls > best_match_theta

    return {
        'seed': seed,
        'per_list': per_list,
        'at_theta_082': {
            'theta': theta_082,
            'hit_rate': float(hit_082),
            'lure_fa_rate': float(lure_fa_082),
            'unrelated_fa_rate': float(unrel_fa_082),
        },
        'best_match': {
            'theta': float(best_match_theta),
            'hit_rate': float(np.mean([s > best_match_theta for s in all_studied_sims])),
            'lure_fa_rate': float(np.mean([s > best_match_theta for s in all_lure_sims])),
            'unrelated_fa_rate': float(np.mean([s > best_match_theta for s in all_unrelated_sims])),
        },
        'operating_curve': operating_curve,
        'overall': {
            'hit_rate': float(hit_082),
            'lure_fa_rate': float(lure_fa_082),
            'unrelated_fa_rate': float(unrel_fa_082),
            'threshold': float(theta_082),
            'n_lists': len(list_names),
        },
    }


def run_all_seeds(architecture, config: dict, seeds: list = None,
                  drm_lists: dict = None) -> dict:
    if seeds is None:
        seeds = config['seeds']
    if drm_lists is None:
        drm_lists = load_drm_word_lists(config)

    per_seed = {}
    for seed in seeds:
        print(f"Running DRM v2 seed={seed} for {architecture.name}...")
        result = run_drm_v2(architecture, config, seed, drm_lists)
        per_seed[str(seed)] = result

    # Aggregate at theta=0.82
    hit_rates = [per_seed[str(s)]['at_theta_082']['hit_rate'] for s in seeds]
    lure_fas = [per_seed[str(s)]['at_theta_082']['lure_fa_rate'] for s in seeds]
    unrel_fas = [per_seed[str(s)]['at_theta_082']['unrelated_fa_rate'] for s in seeds]

    hr_arr, lf_arr, uf_arr = np.array(hit_rates), np.array(lure_fas), np.array(unrel_fas)

    return {
        'architecture': architecture.arch_key,
        'experiment': 'drm',
        'per_seed': per_seed,
        'aggregated': {
            'hit_rate_mean': float(np.mean(hr_arr)),
            'hit_rate_std': float(np.std(hr_arr)),
            'hit_rate_ci': list(bootstrap_confidence_interval(hr_arr)),
            'lure_fa_mean': float(np.mean(lf_arr)),
            'lure_fa_std': float(np.std(lf_arr)),
            'lure_fa_ci': list(bootstrap_confidence_interval(lf_arr)),
            'unrelated_fa_mean': float(np.mean(uf_arr)),
            'unrelated_fa_std': float(np.std(uf_arr)),
            'threshold': 0.82,
        },
    }


def save_results(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'drm.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved DRM v2 results to {path}")
