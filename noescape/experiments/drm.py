"""
Experiment B: DRM False Memory

Tests whether architectures produce false recognition of semantic lures.
Uses the 24 Roediger & McDermott (1995) word lists.

For embedding architectures: cosine similarity of lure to studied-word centroid.
For LLM architectures: ask model if lure word was in studied list.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional

from noescape.utils import set_seed, bootstrap_confidence_interval, load_drm_word_lists


def run_embedding_drm(architecture, config: dict, seed: int, drm_lists: dict = None) -> dict:
    """
    DRM experiment for embedding-based architectures.

    Protocol:
    - Encode each DRM list's studied words
    - Compute centroid of studied embeddings
    - Measure cosine similarity of studied words, lure, and unrelated words to centroid
    - Find threshold where unrelated FA = 0, report lure FA at that threshold
    """
    set_seed(seed)

    if drm_lists is None:
        drm_lists = load_drm_word_lists(config)

    per_list = {}
    all_studied_sims = []
    all_lure_sims = []
    all_unrelated_sims = []

    list_names = list(drm_lists.keys())

    for list_name in list_names:
        data = drm_lists[list_name]
        studied = data['studied']
        lure = data['lure']

        # Get unrelated words from a different list
        other_lists = [k for k in list_names if k != list_name]
        unrelated = drm_lists[other_lists[0]]['studied'][:5]

        # Encode all words
        all_words = studied + [lure] + unrelated
        embeddings = architecture.encode(all_words)

        studied_embs = embeddings[:len(studied)]
        lure_emb = embeddings[len(studied)]
        unrelated_embs = embeddings[len(studied) + 1:]

        # Compute centroid of studied words
        centroid = studied_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        # Cosine similarities to centroid
        studied_sims = [float(np.dot(e / (np.linalg.norm(e) + 1e-8), centroid)) for e in studied_embs]
        lure_sim = float(np.dot(lure_emb / (np.linalg.norm(lure_emb) + 1e-8), centroid))
        unrelated_sims_list = [float(np.dot(e / (np.linalg.norm(e) + 1e-8), centroid)) for e in unrelated_embs]

        all_studied_sims.extend(studied_sims)
        all_lure_sims.append(lure_sim)
        all_unrelated_sims.extend(unrelated_sims_list)

        per_list[list_name] = {
            'studied_mean_sim': float(np.mean(studied_sims)),
            'studied_std_sim': float(np.std(studied_sims)),
            'lure_sim': lure_sim,
            'unrelated_mean_sim': float(np.mean(unrelated_sims_list)),
        }

    # Find threshold where unrelated FA = 0
    unrelated_max = max(all_unrelated_sims)
    theta = unrelated_max + 0.001

    # Compute rates at this threshold
    hit_rate = np.mean([s > theta for s in all_studied_sims])
    lure_fa_rate = np.mean([s > theta for s in all_lure_sims])
    unrelated_fa_rate = np.mean([s > theta for s in all_unrelated_sims])

    # Also compute optimal threshold via sweep
    thetas = np.linspace(0.5, 1.0, 100)
    best_theta = theta
    for t in thetas:
        ufa = np.mean([s > t for s in all_unrelated_sims])
        if ufa == 0:
            best_theta = t
            break

    hit_rate_best = np.mean([s > best_theta for s in all_studied_sims])
    lure_fa_best = np.mean([s > best_theta for s in all_lure_sims])

    # Update per-list with threshold info
    for list_name in list_names:
        ls = per_list[list_name]['lure_sim']
        per_list[list_name]['lure_above_threshold'] = ls > best_theta
        per_list[list_name]['threshold'] = best_theta

    return {
        'seed': seed,
        'per_list': per_list,
        'overall': {
            'hit_rate': float(hit_rate_best),
            'lure_fa_rate': float(lure_fa_best),
            'unrelated_fa_rate': float(np.mean([s > best_theta for s in all_unrelated_sims])),
            'threshold': float(best_theta),
            'n_lists': len(list_names),
        },
        'all_studied_sims': [float(x) for x in all_studied_sims],
        'all_lure_sims': [float(x) for x in all_lure_sims],
        'all_unrelated_sims': [float(x) for x in all_unrelated_sims],
    }


def run_llm_drm(architecture, config: dict, seed: int, drm_lists: dict = None) -> dict:
    """
    DRM experiment for LLM-based architectures.

    Protocol:
    - Present each DRM list to the model
    - Ask about studied words (5), critical lure (1), unrelated words (3)
    - Parse yes/no responses
    """
    set_seed(seed)

    if drm_lists is None:
        drm_lists = load_drm_word_lists(config)

    per_list = {}
    total_hits = 0
    total_studied = 0
    total_lure_fa = 0
    total_lure = 0
    total_unrelated_fa = 0
    total_unrelated = 0

    list_names = list(drm_lists.keys())

    for list_name in list_names:
        data = drm_lists[list_name]
        studied = data['studied']
        lure = data['lure']

        # Select test words
        np.random.shuffle(studied)
        test_studied = studied[:5]
        other_lists = [k for k in list_names if k != list_name]
        unrelated = [drm_lists[other_lists[i]]['studied'][0] for i in range(3)]

        hits = 0
        for word in test_studied:
            if architecture.check_word_in_list(studied, word):
                hits += 1

        lure_endorsed = architecture.check_word_in_list(studied, lure)

        unrelated_endorsed = 0
        for word in unrelated:
            if architecture.check_word_in_list(studied, word):
                unrelated_endorsed += 1

        total_hits += hits
        total_studied += len(test_studied)
        total_lure_fa += int(lure_endorsed)
        total_lure += 1
        total_unrelated_fa += unrelated_endorsed
        total_unrelated += len(unrelated)

        per_list[list_name] = {
            'hit_rate': hits / len(test_studied),
            'lure_fa': int(lure_endorsed),
            'unrelated_fa': unrelated_endorsed / len(unrelated),
        }

    return {
        'seed': seed,
        'per_list': per_list,
        'overall': {
            'hit_rate': total_hits / max(total_studied, 1),
            'lure_fa_rate': total_lure_fa / max(total_lure, 1),
            'unrelated_fa_rate': total_unrelated_fa / max(total_unrelated, 1),
            'n_lists': len(list_names),
        },
    }


def run_experiment(architecture, config: dict, seed: int, drm_lists: dict = None) -> dict:
    """Run DRM experiment appropriate for architecture type."""
    if architecture.arch_key in ('vector_db', 'graph'):
        return run_embedding_drm(architecture, config, seed, drm_lists)
    else:
        return run_llm_drm(architecture, config, seed, drm_lists)


def run_all_seeds(architecture, config: dict, seeds: list = None,
                  drm_lists: dict = None) -> dict:
    """Run across all seeds and aggregate."""
    if seeds is None:
        seeds = config['seeds']

    if drm_lists is None:
        drm_lists = load_drm_word_lists(config)

    per_seed = {}
    for seed in seeds:
        print(f"Running DRM seed={seed} for {architecture.name}...")
        result = run_experiment(architecture, config, seed, drm_lists)
        per_seed[str(seed)] = result

    # Aggregate
    hit_rates = [per_seed[str(s)]['overall']['hit_rate'] for s in seeds]
    lure_fas = [per_seed[str(s)]['overall']['lure_fa_rate'] for s in seeds]
    unrel_fas = [per_seed[str(s)]['overall']['unrelated_fa_rate'] for s in seeds]

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
        },
    }


def save_results(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'drm.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved DRM results to {path}")
