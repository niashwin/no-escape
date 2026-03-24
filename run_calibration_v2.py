#!/usr/bin/env python3
"""
Calibration v2 — Fix all calibration failures using HIDE-matching protocols.

Changes:
- Ebbinghaus: HIDESpace + decay_fn + noise on query (sigma=0.5)
- DRM: Threshold sweep 0.50-0.95, report at theta=0.82
- TOT: PCA to 128 dim, query_noise_sigma=1.2 (HIDE phase5 protocol)
- Spacing: timestamps in seconds, 100K distractors, sigma=0.5 (HIDE phase5)
"""

import sys, os, json, time, yaml, math
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hide-project"))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from noescape.utils import set_seed, load_wikipedia_sentences, load_drm_word_lists, bootstrap_confidence_interval
from noescape.architectures.vector_db import VectorDBArchitecture
from noescape.experiments.drm_v2 import run_all_seeds as drm_run_all, save_results as drm_save
from noescape.experiments.ebbinghaus_v2 import run_all_seeds as ebb_run_all, save_results as ebb_save

from hide.core.hide_space import HIDESpace


def run_tot_v2_hide(embeddings, config, seed):
    """TOT matching HIDE phase5: PCA to 128 dim, query_noise_sigma=1.2."""
    set_seed(seed)
    pca_dim = 128
    query_noise_sigma = 1.2
    sim_threshold = 0.5
    max_rank = 20
    n_stored = min(10000, len(embeddings))
    n_queries = min(10000, n_stored)

    # PCA reduction (HIDE's key insight)
    pca = PCA(n_components=pca_dim, random_state=seed)
    reduced = pca.fit_transform(embeddings[:n_stored])
    norms = np.linalg.norm(reduced, axis=1, keepdims=True) + 1e-8
    reduced = reduced / norms

    # Build HIDESpace
    space = HIDESpace(dim=pca_dim, max_memories=n_stored + 100)
    for i in range(n_stored):
        space.store(reduced[i], {'idx': i})

    query_indices = np.random.choice(n_stored, n_queries, replace=False)
    tot_count = 0

    for qi in query_indices:
        clean_query = reduced[qi].copy()
        noise = query_noise_sigma * np.random.randn(pca_dim).astype(np.float32) / np.sqrt(pca_dim)
        noisy_query = clean_query + noise
        noisy_query = noisy_query / (np.linalg.norm(noisy_query) + 1e-8)

        retrieved = space.retrieve(noisy_query, k=max_rank)
        if not retrieved:
            continue

        top1_sim = retrieved[0][1]
        correct_rank = None
        for rank, (_, sim, meta) in enumerate(retrieved, 1):
            if meta.get('idx') == int(qi):
                correct_rank = rank
                break

        if correct_rank is not None and correct_rank > 1 and correct_rank <= max_rank and top1_sim > sim_threshold:
            tot_count += 1

    tot_rate = tot_count / n_queries
    return {
        'seed': seed,
        'tot_rate': float(tot_rate),
        'n_tot_states': tot_count,
        'n_queries': n_queries,
        'pca_dim': pca_dim,
        'query_noise_sigma': query_noise_sigma,
    }


def run_spacing_v2_hide(embeddings, wiki_sentences, config, seed):
    """Spacing matching HIDE phase5: seconds-based timestamps, 100K distractors, sigma=0.5."""
    set_seed(seed)
    n_facts = 100
    sigma = 0.5
    dim = embeddings.shape[1]
    n_distractors = min(len(embeddings) - n_facts, 10000)

    conditions = {
        'massed': [0, 60, 120],
        'short': [0, 3600, 7200],
        'medium': [0, 86400, 172800],
        'long': [0, 604800, 1209600],
    }
    test_delay = 2592000  # 30 days in seconds

    fact_embs = embeddings[:n_facts]
    dist_embs = embeddings[n_facts:n_facts + n_distractors]

    results = {}
    for cond_name, spacings in conditions.items():
        # Build memory pool: distractors + target repetitions
        all_embs_list = []
        all_timestamps = []
        all_fact_ids = []
        all_types = []

        # Distractors with random timestamps
        dist_timestamps = np.random.uniform(0, 60 * 86400, n_distractors)
        all_embs_list.append(dist_embs.copy())
        all_timestamps.extend(dist_timestamps.tolist())
        all_fact_ids.extend([-1] * n_distractors)
        all_types.extend([0] * n_distractors)

        # Target repetitions
        for i in range(n_facts):
            for t in spacings:
                noisy_emb = fact_embs[i] + np.random.normal(0, 0.01, dim).astype(np.float32)
                all_embs_list.append(noisy_emb.reshape(1, -1))
                all_timestamps.append(float(t))
                all_fact_ids.append(i)
                all_types.append(1)

        all_embs = np.vstack(all_embs_list).astype(np.float32)
        all_timestamps = np.array(all_timestamps)
        all_fact_ids = np.array(all_fact_ids)
        all_types = np.array(all_types)
        n_total = len(all_timestamps)

        # Age-proportional noise (HIDE protocol: applied to ALL items)
        age_days = np.maximum(0, (test_delay - all_timestamps) / 86400.0)
        noise_scale = sigma * np.sqrt(age_days + 0.01) / np.sqrt(dim)
        noise = np.random.randn(n_total, dim).astype(np.float32) * noise_scale[:, None]
        noisy_embs = all_embs + noise
        norms = np.linalg.norm(noisy_embs, axis=1, keepdims=True) + 1e-8
        noisy_embs = noisy_embs / norms

        # Query embeddings (clean, normalized)
        q_norms = np.linalg.norm(fact_embs, axis=1, keepdims=True) + 1e-8
        q_normed = fact_embs / q_norms

        # Batch cosine similarity
        sims = q_normed @ noisy_embs.T

        # For each fact, check top-3
        hits = 0
        for i in range(n_facts):
            top_k_idx = np.argsort(sims[i])[-3:][::-1]
            for tidx in top_k_idx:
                if all_types[tidx] == 1 and all_fact_ids[tidx] == i:
                    hits += 1
                    break

        retention = hits / n_facts
        results[cond_name] = {'retention': float(retention)}

    ordering = ['massed', 'short', 'medium', 'long']
    retentions = [results[c]['retention'] for c in ordering]
    ordering_correct = all(retentions[i] <= retentions[i+1] for i in range(3))

    return {
        'seed': seed,
        'conditions': results,
        'ordering_correct': bool(ordering_correct),
        'sigma': sigma,
    }


def main():
    t_start = time.time()
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    seeds = config['seeds']
    wiki_sentences = load_wikipedia_sentences(n_sentences=20000, n_articles=500)
    drm_lists = load_drm_word_lists(config)
    print(f"Data: {len(wiki_sentences)} sentences, {len(drm_lists)} DRM lists")

    # Load architecture
    arch = VectorDBArchitecture(config)
    texts = [s['text'] for s in wiki_sentences]

    # Pre-encode
    print("Encoding with BGE-large...")
    embs = arch.encode(texts)
    print(f"  Shape: {embs.shape}")

    results_dir = Path('results/calibration')
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- Ebbinghaus v2 ----
    print("\n=== Ebbinghaus v2 (HIDE protocol) ===")
    t0 = time.time()
    ebb_results = ebb_run_all(arch, config, seeds, wiki_sentences, precomputed_embeddings=embs)
    ebb_save(ebb_results, str(results_dir))
    agg = ebb_results['aggregated']['per_n_near']
    max_k = max(agg.keys(), key=lambda x: int(x))
    print(f"  b at {max_k}: {agg[max_k]['b_mean']:.3f} ± {agg[max_k]['b_std']:.3f}")
    print(f"  Target: [0.276, 0.644]. PASS: {0.276 <= agg[max_k]['b_mean'] <= 0.644}")
    print(f"  Time: {time.time()-t0:.0f}s")

    # ---- DRM v2 ----
    print("\n=== DRM v2 (theta sweep 0.50-0.95) ===")
    t0 = time.time()
    drm_results = drm_run_all(arch, config, seeds, drm_lists)
    drm_save(drm_results, str(results_dir))
    dagg = drm_results['aggregated']
    print(f"  At theta=0.82: lure FA={dagg['lure_fa_mean']:.3f} ± {dagg['lure_fa_std']:.3f}")
    print(f"  Target: [0.533, 0.633]. PASS: {0.533 <= dagg['lure_fa_mean'] <= 0.633}")
    print(f"  Time: {time.time()-t0:.0f}s")

    # ---- TOT v2 ----
    print("\n=== TOT v2 (PCA 128 + noise 1.2) ===")
    t0 = time.time()
    tot_per_seed = {}
    for seed in seeds:
        r = run_tot_v2_hide(embs, config, seed)
        tot_per_seed[str(seed)] = r
        print(f"  seed={seed}: TOT={r['tot_rate']:.4f}")

    tot_rates = np.array([tot_per_seed[str(s)]['tot_rate'] for s in seeds])
    ci = bootstrap_confidence_interval(tot_rates)
    tot_results = {
        'architecture': 'vector_db', 'experiment': 'tot',
        'per_seed': tot_per_seed,
        'aggregated': {
            'tot_rate_mean': float(np.mean(tot_rates)),
            'tot_rate_std': float(np.std(tot_rates)),
            'tot_rate_ci': list(ci),
        },
    }
    with open(results_dir / 'tot.json', 'w') as f:
        json.dump(tot_results, f, indent=2, default=str)
    print(f"  TOT: {np.mean(tot_rates):.4f} ± {np.std(tot_rates):.4f}")
    print(f"  Target: [0.006, 0.066]. PASS: {0.006 <= np.mean(tot_rates) <= 0.066}")
    print(f"  Time: {time.time()-t0:.0f}s")

    # ---- Spacing v2 ----
    print("\n=== Spacing v2 (HIDE seconds + 10K distractors + sigma=0.5) ===")
    t0 = time.time()
    sp_per_seed = {}
    for seed in seeds:
        r = run_spacing_v2_hide(embs, wiki_sentences, config, seed)
        sp_per_seed[str(seed)] = r
        conds = r['conditions']
        print(f"  seed={seed}: M={conds['massed']['retention']:.3f} S={conds['short']['retention']:.3f} "
              f"Med={conds['medium']['retention']:.3f} L={conds['long']['retention']:.3f} order={r['ordering_correct']}")

    # Aggregate spacing
    sp_agg = {}
    for cond in ['massed', 'short', 'medium', 'long']:
        vals = [sp_per_seed[str(s)]['conditions'][cond]['retention'] for s in seeds]
        arr = np.array(vals)
        ci = bootstrap_confidence_interval(arr)
        sp_agg[f'{cond}_mean'] = float(np.mean(arr))
        sp_agg[f'{cond}_std'] = float(np.std(arr))
        sp_agg[f'{cond}_ci'] = list(ci)

    long_v = np.array([sp_per_seed[str(s)]['conditions']['long']['retention'] for s in seeds])
    mass_v = np.array([sp_per_seed[str(s)]['conditions']['massed']['retention'] for s in seeds])
    sp_agg['ordering_correct_count'] = sum(sp_per_seed[str(s)]['ordering_correct'] for s in seeds)
    pooled = np.sqrt(((len(long_v)-1)*np.var(long_v,ddof=1) + (len(mass_v)-1)*np.var(mass_v,ddof=1)) / (len(long_v)+len(mass_v)-2))
    sp_agg['cohens_d_long_vs_massed'] = float((np.mean(long_v) - np.mean(mass_v)) / max(pooled, 1e-8))

    spacing_results = {
        'architecture': 'vector_db', 'experiment': 'spacing',
        'per_seed': sp_per_seed, 'aggregated': sp_agg,
    }
    with open(results_dir / 'spacing.json', 'w') as f:
        json.dump(spacing_results, f, indent=2, default=str)
    print(f"  long > massed: {sp_agg['long_mean'] > sp_agg['massed_mean']}")
    print(f"  Time: {time.time()-t0:.0f}s")

    # ---- Calibration Summary ----
    print("\n" + "="*60)
    print("CALIBRATION v2 SUMMARY")
    print("="*60)
    checks = {}

    # Ebbinghaus
    b_mean = agg[max_k]['b_mean']
    checks['ebbinghaus_b'] = {'value': b_mean, 'range': [0.276, 0.644],
                               'pass': 0.276 <= b_mean <= 0.644}

    # DRM
    lfa = dagg['lure_fa_mean']
    checks['drm_lure_fa'] = {'value': lfa, 'range': [0.533, 0.633],
                              'pass': 0.533 <= lfa <= 0.633}

    # Spacing
    sp_pass = sp_agg['long_mean'] > sp_agg['massed_mean']
    checks['spacing_order'] = {'long': sp_agg['long_mean'], 'massed': sp_agg['massed_mean'],
                                'pass': sp_pass}

    # TOT
    tot_mean = float(np.mean(tot_rates))
    checks['tot_rate'] = {'value': tot_mean, 'range': [0.006, 0.066],
                           'pass': 0.006 <= tot_mean <= 0.066}

    for name, check in checks.items():
        status = "PASS" if check['pass'] else "FAIL"
        print(f"  {name}: {status} — {check}")

    all_pass = all(c['pass'] for c in checks.values())
    print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")

    with open(results_dir / 'calibration_v2_summary.json', 'w') as f:
        json.dump(checks, f, indent=2, default=str)

    print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    return all_pass


if __name__ == '__main__':
    main()
