"""
Experiment A: Ebbinghaus Forgetting Curves

For embedding-based architectures (1, 4): encode target + competitors with
age-proportional noise, measure retrieval accuracy vs. memory age.

For LLM-based architectures (2, 3, 5): measure answer accuracy with varying
numbers of competing facts in context/store.

Fits power-law forgetting curves and extracts the forgetting exponent b.
"""

import numpy as np
import json
import os
import time
from typing import Dict, List, Optional
from scipy.optimize import curve_fit
from pathlib import Path

from noescape.utils import (
    set_seed, bootstrap_confidence_interval, fit_forgetting_curve,
    load_wikipedia_sentences
)


def _power_law_3param(t, a, b, c):
    return a * np.power(t + 1e-6, -b) + c


def run_embedding_ebbinghaus(architecture, config: dict, seed: int,
                              wiki_sentences: List[Dict] = None,
                              precomputed_embeddings: np.ndarray = None,
                              precomputed_article_ids: np.ndarray = None) -> dict:
    """
    Run Ebbinghaus experiment for embedding-based architectures.

    OPTIMIZED: Pre-encode all sentences once, reuse embeddings across trials.
    """
    set_seed(seed)

    arch_cfg = config['architectures'].get(architecture.arch_key, {})
    exp_cfg = config['experiments']['ebbinghaus']
    n_near_values = arch_cfg.get('near_competitor_counts',
                                  exp_cfg.get('default_n_near', [0, 10, 50, 100, 500, 1000, 5000, 10000]))
    n_target_facts = exp_cfg.get('n_target_facts', 100)
    noise_sigma = arch_cfg.get('noise_sigma',
                                config['architectures'].get('vector_db', {}).get('noise_sigma', 0.25))
    simulated_days = exp_cfg.get('simulated_days', 30)
    temporal_bins = exp_cfg.get('temporal_bins', 10)

    if wiki_sentences is None:
        wiki_sentences = load_wikipedia_sentences(n_sentences=20000, n_articles=500)

    # Pre-encode ALL sentences once (the key optimization)
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

    results_per_n_near = {}
    age_bins = np.linspace(0.1, simulated_days, temporal_bins)

    for n_near in n_near_values:
        print(f"  [seed={seed}] n_near={n_near}...")
        t0 = time.time()

        per_bin_correct = np.zeros(temporal_bins)
        per_bin_total = np.zeros(temporal_bins)

        n_trials = min(n_target_facts, len(unique_articles))

        for trial in range(n_trials):
            # Pick target from an article
            target_aid = unique_articles[trial % len(unique_articles)]
            target_mask = article_ids == target_aid
            target_indices = np.where(target_mask)[0]
            if len(target_indices) == 0:
                continue
            target_idx = target_indices[trial % len(target_indices)]
            target_emb = all_embeddings[target_idx]

            # Near competitors: same article first, then similar articles
            near_indices = []
            # Same article (excluding target)
            same_article = target_indices[target_indices != target_idx]
            near_indices.extend(same_article.tolist())

            # If need more, add from other articles
            if len(near_indices) < n_near:
                other_mask = ~target_mask
                other_indices = np.where(other_mask)[0]
                if len(other_indices) > 0:
                    # Sample from similar articles (nearby in sequence = similar topic)
                    needed = n_near - len(near_indices)
                    if needed > len(other_indices):
                        near_indices.extend(other_indices.tolist())
                    else:
                        sampled = np.random.choice(other_indices, needed, replace=False)
                        near_indices.extend(sampled.tolist())

            near_indices = near_indices[:n_near]

            if not near_indices:
                competitor_embs = np.zeros((0, dim))
            else:
                competitor_embs = all_embeddings[np.array(near_indices)]

            # Test at different ages
            for bin_idx, age in enumerate(age_bins):
                noise_scale = noise_sigma * np.sqrt(age + 0.01) / np.sqrt(dim)
                noisy_target = target_emb + noise_scale * np.random.randn(dim)
                noisy_target = noisy_target / (np.linalg.norm(noisy_target) + 1e-8)

                if competitor_embs.shape[0] > 0:
                    # Add noise to competitors
                    comp_ages = np.random.uniform(0, simulated_days, competitor_embs.shape[0])
                    comp_noise_scales = noise_sigma * np.sqrt(comp_ages + 0.01) / np.sqrt(dim)
                    noise = np.random.randn(*competitor_embs.shape)
                    noisy_comps = competitor_embs + noise * comp_noise_scales[:, None]
                    norms = np.linalg.norm(noisy_comps, axis=1, keepdims=True)
                    noisy_comps = noisy_comps / (norms + 1e-8)

                    all_embs = np.vstack([noisy_target.reshape(1, -1), noisy_comps])
                else:
                    all_embs = noisy_target.reshape(1, -1)

                query_emb = target_emb / (np.linalg.norm(target_emb) + 1e-8)
                sims = all_embs @ query_emb
                retrieved_idx = np.argmax(sims)

                per_bin_correct[bin_idx] += int(retrieved_idx == 0)
                per_bin_total[bin_idx] += 1

        # Compute accuracy per bin
        valid_bins = per_bin_total > 0
        ages = age_bins[valid_bins]
        accuracies = per_bin_correct[valid_bins] / per_bin_total[valid_bins]

        fit_result = fit_forgetting_curve(ages, accuracies)

        results_per_n_near[str(n_near)] = {
            'ages': ages.tolist(),
            'accuracies': accuracies.tolist(),
            'fitted_b': fit_result['b'],
            'fitted_a': fit_result['a'],
            'fitted_c': fit_result['c'],
            'r_squared': fit_result['r_squared'],
            'fit_success': fit_result['fit_success'],
        }
        print(f"    n_near={n_near}: b={fit_result['b']:.3f}, R²={fit_result['r_squared']:.3f} ({time.time()-t0:.1f}s)")

    return {
        'seed': seed,
        'n_near_values': [int(x) for x in n_near_values],
        'per_n_near': results_per_n_near,
    }


def run_llm_ebbinghaus(architecture, config: dict, seed: int,
                        wiki_sentences: List[Dict] = None) -> dict:
    """
    Run Ebbinghaus experiment for LLM-based architectures.
    Uses position in context as proxy for age.
    """
    set_seed(seed)

    arch_cfg = config['architectures'].get(architecture.arch_key, {})
    exp_cfg = config['experiments']['ebbinghaus']
    n_near_values = arch_cfg.get('near_competitor_counts',
                                  exp_cfg.get('default_n_near', [0, 10, 50, 100, 500]))
    n_target_facts = arch_cfg.get('n_target_facts', min(exp_cfg.get('n_target_facts', 50), 30))

    if wiki_sentences is None:
        wiki_sentences = load_wikipedia_sentences(n_sentences=5000, n_articles=200)

    articles = {}
    for s in wiki_sentences:
        aid = s['article_id']
        if aid not in articles:
            articles[aid] = []
        articles[aid].append(s)

    article_ids = list(articles.keys())
    np.random.shuffle(article_ids)

    results_per_n_near = {}
    temporal_bins = 5

    for n_near in n_near_values:
        print(f"  [seed={seed}] n_near={n_near}...")
        per_bin_correct = np.zeros(temporal_bins)
        per_bin_total = np.zeros(temporal_bins)

        n_trials = min(n_target_facts, 20)

        for trial in range(n_trials):
            target_aid = article_ids[trial % len(article_ids)]
            target_sents = articles[target_aid]
            if len(target_sents) < 2:
                continue

            target_text = target_sents[0]['text']
            question = f"What does the following fact state: {target_text[:50]}...?"

            competitors = []
            for s in target_sents[1:]:
                competitors.append(s['text'])
            for aid in article_ids:
                if aid != target_aid:
                    for s in articles[aid][:3]:
                        competitors.append(s['text'])
                if len(competitors) >= n_near:
                    break
            competitors = competitors[:n_near]

            total_facts = 1 + len(competitors)
            for bin_idx in range(temporal_bins):
                target_pos = int((bin_idx / max(temporal_bins - 1, 1)) * max(total_facts - 1, 0))
                fact_list = list(competitors)
                fact_list.insert(min(target_pos, len(fact_list)), target_text)

                if hasattr(architecture, 'answer_question_in_context'):
                    answer = architecture.answer_question_in_context(fact_list[:500], question)
                    target_words = set(target_text.lower().split())
                    answer_words = set(answer.lower().split())
                    stopwords = {'the', 'a', 'an', 'is', 'was', 'are', 'of', 'in', 'to', 'and', 'that', 'it'}
                    overlap = len((target_words - stopwords) & (answer_words - stopwords))
                    correct = overlap >= 2
                else:
                    architecture.clear()
                    architecture.store(fact_list[:500])
                    results = architecture.retrieve(question, top_k=1)
                    correct = bool(results and fact_list[results[0][0]] == target_text)

                per_bin_correct[bin_idx] += int(correct)
                per_bin_total[bin_idx] += 1

        simulated_days = 30
        valid = per_bin_total > 0
        ages = np.array([(i+1)/temporal_bins * simulated_days for i in range(temporal_bins)])[valid]
        accuracies = (per_bin_correct / np.maximum(per_bin_total, 1))[valid]

        fit_result = fit_forgetting_curve(ages, accuracies)

        results_per_n_near[str(n_near)] = {
            'ages': ages.tolist(),
            'accuracies': accuracies.tolist(),
            'fitted_b': fit_result['b'],
            'fitted_a': fit_result['a'],
            'fitted_c': fit_result['c'],
            'r_squared': fit_result['r_squared'],
            'fit_success': fit_result['fit_success'],
        }

    return {
        'seed': seed,
        'n_near_values': [int(x) for x in n_near_values],
        'per_n_near': results_per_n_near,
    }


def run_experiment(architecture, config: dict, seed: int,
                   wiki_sentences: List[Dict] = None,
                   precomputed_embeddings: np.ndarray = None) -> dict:
    if architecture.arch_key in ('vector_db', 'graph'):
        return run_embedding_ebbinghaus(architecture, config, seed, wiki_sentences,
                                         precomputed_embeddings=precomputed_embeddings)
    else:
        return run_llm_ebbinghaus(architecture, config, seed, wiki_sentences)


def run_all_seeds(architecture, config: dict, seeds: list = None,
                  wiki_sentences: List[Dict] = None,
                  precomputed_embeddings: np.ndarray = None) -> dict:
    if seeds is None:
        seeds = config['seeds']

    # Pre-encode once for embedding architectures
    if architecture.arch_key in ('vector_db', 'graph') and precomputed_embeddings is None:
        if wiki_sentences is None:
            wiki_sentences = load_wikipedia_sentences(n_sentences=20000, n_articles=500)
        texts = [s['text'] for s in wiki_sentences]
        print(f"  Pre-encoding {len(texts)} sentences for all seeds...")
        precomputed_embeddings = architecture.encode(texts)
        print(f"  Encoding complete: shape={precomputed_embeddings.shape}")

    per_seed = {}
    for seed in seeds:
        print(f"Running Ebbinghaus seed={seed} for {architecture.name}...")
        result = run_experiment(architecture, config, seed, wiki_sentences,
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
        b_ci = bootstrap_confidence_interval(b_arr) if len(b_arr) > 1 else (b_arr[0], b_arr[0])

        aggregated['per_n_near'][n_near_str] = {
            'b_mean': float(np.mean(b_arr)),
            'b_std': float(np.std(b_arr)),
            'b_ci_lower': b_ci[0],
            'b_ci_upper': b_ci[1],
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
    print(f"Saved Ebbinghaus results to {path}")
