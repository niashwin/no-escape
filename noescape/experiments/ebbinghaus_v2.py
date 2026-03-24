"""
Ebbinghaus Experiment v2 — Matches HIDE protocol exactly.

Key differences from v1:
- Uses HIDESpace with decay_fn during retrieval (power-law decay)
- Adds age-proportional noise to QUERY, not stored items
- Stores targets + distractors in same HIDESpace
- Sweeps sigma and beta to find best config
- Uses HIDE's fit_power_law (2-param: R(t) = a * t^(-b))
"""

import numpy as np
import json
import os
import sys
import time
from typing import Dict, List, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "hide-project"))

from hide.core.hide_space import HIDESpace
from hide.core.interference import age_proportional_noise, fit_forgetting_curve
from hide.utils.metrics import bootstrap_ci, fit_power_law

from noescape.utils import set_seed, load_wikipedia_sentences, bootstrap_confidence_interval


def run_ebbinghaus_v2(architecture, config: dict, seed: int,
                       wiki_sentences: List[Dict] = None,
                       precomputed_embeddings: np.ndarray = None) -> dict:
    """
    Run Ebbinghaus experiment matching HIDE Phase 2 protocol.

    Protocol:
    - Store n_near competitor embeddings + target in HIDESpace
    - Apply power-law decay during retrieval: S(t) = (1 + beta*t)^(-psi)
    - Add age-proportional noise to query embedding
    - Measure top-5 retrieval accuracy across age bins
    - Fit 2-param power law: R(t) = a * t^(-b)
    """
    set_seed(seed)

    exp_cfg = config['experiments']['ebbinghaus']
    arch_cfg = config['architectures'].get(architecture.arch_key, {})
    n_near_values = arch_cfg.get('near_competitor_counts',
                                  exp_cfg.get('default_n_near', [0, 10, 50, 100, 500, 1000, 5000, 10000]))
    n_target_facts = exp_cfg.get('n_target_facts', 100)
    simulated_days = exp_cfg.get('simulated_days', 30)
    temporal_bins = exp_cfg.get('temporal_bins', 10)

    # HIDE parameters
    noise_sigma = 0.5  # HIDE's best_sigma from phase5.yaml
    decay_beta = arch_cfg.get('decay_beta', 1.0)
    decay_psi = arch_cfg.get('decay_psi', 0.5)

    if wiki_sentences is None:
        wiki_sentences = load_wikipedia_sentences(n_sentences=20000, n_articles=500)

    texts = [s['text'] for s in wiki_sentences]
    article_ids = np.array([s['article_id'] for s in wiki_sentences])

    if precomputed_embeddings is not None:
        all_embeddings = precomputed_embeddings
    else:
        print(f"    Encoding {len(texts)} sentences...")
        all_embeddings = architecture.encode(texts)

    dim = all_embeddings.shape[1]
    unique_articles = np.unique(article_ids)
    np.random.shuffle(unique_articles)

    # Time setup
    total_seconds = simulated_days * 86400
    query_time = total_seconds

    # Age bins (in days)
    age_bins_days = np.linspace(0.5, simulated_days - 0.5, temporal_bins)

    results_per_n_near = {}

    for n_near in n_near_values:
        print(f"  [seed={seed}] n_near={n_near}...")
        t0 = time.time()

        per_bin_correct = np.zeros(temporal_bins)
        per_bin_total = np.zeros(temporal_bins)

        n_trials = min(n_target_facts, len(unique_articles))

        for trial in range(n_trials):
            # Pick target
            target_aid = unique_articles[trial % len(unique_articles)]
            target_mask = article_ids == target_aid
            target_indices = np.where(target_mask)[0]
            if len(target_indices) == 0:
                continue
            target_idx = target_indices[trial % len(target_indices)]
            target_emb = all_embeddings[target_idx]

            # Random timestamp for target
            target_timestamp = np.random.uniform(0, total_seconds)
            target_age_days = (query_time - target_timestamp) / 86400.0

            # Find which age bin this falls in
            bin_idx = int(target_age_days / simulated_days * temporal_bins)
            bin_idx = min(bin_idx, temporal_bins - 1)

            # Build HIDESpace with distractors + target
            space = HIDESpace(dim=dim, max_memories=n_near + 10)

            # Store distractors
            if n_near > 0:
                other_indices = np.where(~target_mask)[0]
                if len(other_indices) > n_near:
                    dist_indices = np.random.choice(other_indices, n_near, replace=False)
                else:
                    dist_indices = other_indices
                dist_timestamps = np.random.uniform(0, total_seconds, len(dist_indices))

                for di, dt in zip(dist_indices, dist_timestamps):
                    space.store(all_embeddings[di], {
                        'type': 'distractor',
                        'timestamp': float(dt),
                        'idx': int(di),
                    })

            # Store target
            target_memory_id = space.store(target_emb, {
                'type': 'target',
                'timestamp': float(target_timestamp),
                'idx': int(target_idx),
                'text': texts[target_idx][:100],
            })

            # Query with noise
            noise = noise_sigma * np.sqrt(target_age_days + 0.01) / np.sqrt(dim)
            noisy_query = target_emb + noise * np.random.randn(dim).astype(np.float32)
            noisy_query = noisy_query / (np.linalg.norm(noisy_query) + 1e-8)

            # Decay function
            def decay_fn(meta, _beta=decay_beta, _psi=decay_psi):
                ts = meta.get('timestamp', 0.0)
                dt_days = abs(query_time - ts) / 86400.0
                return (1.0 + _beta * dt_days) ** (-_psi)

            # Retrieve with decay
            retrieved = space.retrieve(noisy_query, k=5, decay_fn=decay_fn, query_time=query_time)

            # Check if target is in top-5
            found = False
            for _, sim, meta in retrieved:
                if meta.get('type') == 'target' and meta.get('idx') == int(target_idx):
                    found = True
                    break

            per_bin_correct[bin_idx] += int(found)
            per_bin_total[bin_idx] += 1

        # Compute accuracy per bin
        valid = per_bin_total > 0
        ages = age_bins_days[valid]
        accuracies = per_bin_correct[valid] / per_bin_total[valid]

        # Fit 2-param power law (HIDE uses R(t) = a * t^(-b))
        fit_result = {'a': 0, 'b': 0, 'r_squared': 0, 'fit_success': False}
        if len(ages) >= 3:
            try:
                fit_dict = fit_power_law(ages, accuracies)
                fit_result = {
                    'a': float(fit_dict.get('a', 0)),
                    'b': float(fit_dict.get('b', 0)),
                    'r_squared': float(fit_dict.get('r_squared', 0)),
                    'fit_success': True,
                }
            except Exception:
                # Fallback to our own fitter
                from noescape.utils import fit_forgetting_curve as noescape_fit
                fr = noescape_fit(ages, accuracies)
                fit_result = {
                    'a': fr['a'], 'b': fr['b'], 'c': fr.get('c', 0),
                    'r_squared': fr['r_squared'], 'fit_success': fr['fit_success'],
                }

        results_per_n_near[str(n_near)] = {
            'ages': ages.tolist(),
            'accuracies': accuracies.tolist(),
            'fitted_b': fit_result['b'],
            'fitted_a': fit_result['a'],
            'r_squared': fit_result['r_squared'],
            'fit_success': fit_result.get('fit_success', True),
        }
        elapsed = time.time() - t0
        print(f"    n_near={n_near}: b={fit_result['b']:.3f}, R²={fit_result['r_squared']:.3f} ({elapsed:.1f}s)")

    return {
        'seed': seed,
        'n_near_values': [int(x) for x in n_near_values],
        'per_n_near': results_per_n_near,
        'noise_sigma': noise_sigma,
        'decay_beta': decay_beta,
        'decay_psi': decay_psi,
    }


def run_all_seeds(architecture, config: dict, seeds: list = None,
                  wiki_sentences: List[Dict] = None,
                  precomputed_embeddings: np.ndarray = None) -> dict:
    if seeds is None:
        seeds = config['seeds']

    # Pre-encode once
    if architecture.arch_key in ('vector_db', 'graph') and precomputed_embeddings is None:
        if wiki_sentences is None:
            wiki_sentences = load_wikipedia_sentences(n_sentences=20000, n_articles=500)
        texts = [s['text'] for s in wiki_sentences]
        print(f"  Pre-encoding {len(texts)} sentences...")
        precomputed_embeddings = architecture.encode(texts)

    per_seed = {}
    for seed in seeds:
        print(f"Running Ebbinghaus v2 seed={seed} for {architecture.name}...")
        result = run_ebbinghaus_v2(architecture, config, seed, wiki_sentences,
                                    precomputed_embeddings=precomputed_embeddings)
        per_seed[str(seed)] = result

    # Aggregate
    first_seed = per_seed[str(seeds[0])]
    aggregated = {'per_n_near': {}}

    for n_near_str in first_seed['per_n_near']:
        b_values = []
        r2_values = []
        for s in seeds:
            sr = per_seed[str(s)]
            if n_near_str in sr['per_n_near']:
                b_values.append(sr['per_n_near'][n_near_str]['fitted_b'])
                r2_values.append(sr['per_n_near'][n_near_str]['r_squared'])

        b_arr = np.array(b_values)
        r2_arr = np.array(r2_values)
        ci = bootstrap_confidence_interval(b_arr) if len(b_arr) > 1 else (b_arr[0], b_arr[0])

        aggregated['per_n_near'][n_near_str] = {
            'b_mean': float(np.mean(b_arr)),
            'b_std': float(np.std(b_arr)),
            'b_ci_lower': ci[0],
            'b_ci_upper': ci[1],
            'r2_mean': float(np.mean(r2_arr)),
            'r2_std': float(np.std(r2_arr)),
            'n_seeds': len(b_values),
        }

    return {
        'architecture': architecture.arch_key,
        'experiment': 'ebbinghaus',
        'per_seed': per_seed,
        'aggregated': aggregated,
    }


def save_results(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'ebbinghaus.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved Ebbinghaus v2 results to {path}")
