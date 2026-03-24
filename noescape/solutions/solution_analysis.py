"""
Solution Analysis — Section 5: Proving No-Escape Empirically

Four proposed solutions, each tested to show the immunity-vs-usefulness tradeoff:
1. High nominal dimensionality (zero-pad to increase d_nominal)
2. BM25 keyword retrieval (exact match eliminates false recall)
3. Perfect orthogonalization (Gram-Schmidt on stored embeddings)
4. Memory compression (cluster and merge similar memories)
"""

import numpy as np
import json
import os
from typing import Dict, List
from scipy.optimize import curve_fit
from pathlib import Path

from noescape.utils import set_seed, bootstrap_confidence_interval, fit_forgetting_curve


def solution1_high_dimensionality(config: dict, encode_fn, wiki_sentences: list) -> Dict:
    """
    Solution 1: Increase nominal dimensionality.

    Test: PCA to lower dims, zero-pad to higher dims.
    Expected: b depends on d_eff not d_nominal.
    """
    set_seed(42)
    sol_cfg = config['solutions']['high_dim']
    d_values = sol_cfg.get('d_values', [64, 128, 256, 512, 1024, 2048, 4096])
    n_competitors = sol_cfg.get('n_competitors', 10000)

    texts = [s['text'] for s in wiki_sentences[:5000]]
    base_embeddings = encode_fn(texts)
    dim_orig = base_embeddings.shape[1]

    results = {}
    noise_sigma = 0.25

    for d_target in d_values:
        if d_target < dim_orig:
            # PCA reduction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=d_target)
            embeddings = pca.fit_transform(base_embeddings)
            # Re-normalize
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        elif d_target > dim_orig:
            # Zero-pad
            pad = np.zeros((base_embeddings.shape[0], d_target - dim_orig))
            embeddings = np.hstack([base_embeddings, pad])
            embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        else:
            embeddings = base_embeddings.copy()

        # Compute d_eff
        from noescape.utils import compute_participation_ratio
        pr = compute_participation_ratio(embeddings[:2000])
        d_eff = pr['d_eff']

        # Run mini Ebbinghaus at n_competitors
        dim = embeddings.shape[1]
        n_trials = 100
        age_bins = np.linspace(0.1, 30, 10)

        per_bin_correct = np.zeros(len(age_bins))
        per_bin_total = np.zeros(len(age_bins))

        for trial in range(n_trials):
            target_idx = trial % len(embeddings)
            target = embeddings[target_idx]

            # Competitors
            comp_indices = np.random.choice(len(embeddings), min(n_competitors, len(embeddings) - 1), replace=False)
            comp_indices = comp_indices[comp_indices != target_idx][:n_competitors]
            competitors = embeddings[comp_indices]

            for bi, age in enumerate(age_bins):
                noise_scale = noise_sigma * np.sqrt(age + 0.01) / np.sqrt(dim)
                noisy_target = target + noise_scale * np.random.randn(dim)
                noisy_target = noisy_target / (np.linalg.norm(noisy_target) + 1e-8)

                all_embs = np.vstack([noisy_target.reshape(1, -1), competitors])
                query = target / (np.linalg.norm(target) + 1e-8)
                sims = all_embs @ query
                per_bin_correct[bi] += int(np.argmax(sims) == 0)
                per_bin_total[bi] += 1

        accuracies = per_bin_correct / np.maximum(per_bin_total, 1)
        fit = fit_forgetting_curve(age_bins, accuracies)

        results[str(d_target)] = {
            'd_nominal': d_target,
            'd_eff': float(d_eff),
            'fitted_b': fit['b'],
            'r_squared': fit['r_squared'],
            'method': 'pca' if d_target < dim_orig else ('zeropad' if d_target > dim_orig else 'original'),
        }

    return results


def solution2_bm25_keyword(config: dict, wiki_sentences: list, encode_fn) -> Dict:
    """
    Solution 2: BM25 keyword retrieval.

    Test: BM25 eliminates DRM false recall but fails on paraphrase retrieval.
    """
    set_seed(42)
    from rank_bm25 import BM25Okapi

    texts = [s['text'] for s in wiki_sentences[:5000]]
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    # Test DRM with BM25
    from noescape.utils import load_drm_word_lists
    drm_lists = load_drm_word_lists(config)

    drm_hits = 0
    drm_lure_fa = 0
    drm_unrelated_fa = 0
    drm_total_studied = 0
    drm_total_lure = 0
    drm_total_unrelated = 0

    for list_name, data in drm_lists.items():
        studied = data['studied']
        lure = data['lure']
        unrelated = list(drm_lists.values())[0]['studied'][:3]

        # Store studied words
        corpus = studied.copy()
        tok_corpus = [w.lower().split() for w in corpus]
        list_bm25 = BM25Okapi(tok_corpus)

        # Test studied
        for word in studied[:5]:
            scores = list_bm25.get_scores(word.lower().split())
            if max(scores) > 0:
                drm_hits += 1
            drm_total_studied += 1

        # Test lure (exact match should fail)
        scores = list_bm25.get_scores(lure.lower().split())
        if max(scores) > 0 and lure.lower() in [w.lower() for w in corpus]:
            drm_lure_fa += 1
        drm_total_lure += 1

        # Test unrelated
        for word in unrelated:
            scores = list_bm25.get_scores(word.lower().split())
            if max(scores) > 0 and word.lower() in [w.lower() for w in corpus]:
                drm_unrelated_fa += 1
            drm_total_unrelated += 1

    # Test semantic retrieval accuracy (paraphrase benchmark)
    # Use embedding-based retrieval as ground truth
    semantic_embeddings = encode_fn(texts[:1000])

    n_test = 200
    bm25_correct = 0
    semantic_correct = 0

    for i in range(n_test):
        query_idx = i
        # Create paraphrase-like query (use embedding similarity to find near-duplicates)
        query = texts[query_idx]

        # BM25 retrieval
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_scores[query_idx] = -1  # exclude self
        bm25_top = np.argmax(bm25_scores)

        # Semantic retrieval
        sims = semantic_embeddings @ semantic_embeddings[query_idx]
        sims[query_idx] = -1
        sem_top = np.argmax(sims)

        # Check if they agree (semantic is ground truth)
        if bm25_top == sem_top:
            bm25_correct += 1
        semantic_correct += 1

    return {
        'drm': {
            'hit_rate': drm_hits / max(drm_total_studied, 1),
            'lure_fa_rate': drm_lure_fa / max(drm_total_lure, 1),
            'unrelated_fa_rate': drm_unrelated_fa / max(drm_total_unrelated, 1),
        },
        'semantic_accuracy': {
            'bm25_agrees_with_semantic': bm25_correct / max(n_test, 1),
            'n_test': n_test,
        },
        'tradeoff': {
            'immunity': 1.0 - drm_lure_fa / max(drm_total_lure, 1),
            'usefulness': bm25_correct / max(n_test, 1),
        },
    }


def solution3_orthogonalization(config: dict, encode_fn, wiki_sentences: list) -> Dict:
    """
    Solution 3: Orthogonalize stored embeddings (Gram-Schmidt / random projection).

    Test: eliminates interference but destroys semantic retrieval.
    """
    set_seed(42)
    texts = [s['text'] for s in wiki_sentences[:2000]]
    embeddings = encode_fn(texts)
    dim = embeddings.shape[1]

    results = {}

    # Method 1: Gram-Schmidt (on subset - full GS impractical for >dim vectors)
    n_gs = min(dim, len(embeddings))
    gs_embs = embeddings[:n_gs].copy()

    # Modified Gram-Schmidt
    for i in range(1, n_gs):
        for j in range(i):
            proj = np.dot(gs_embs[i], gs_embs[j]) / (np.dot(gs_embs[j], gs_embs[j]) + 1e-8)
            gs_embs[i] -= proj * gs_embs[j]
        norm = np.linalg.norm(gs_embs[i])
        if norm > 1e-8:
            gs_embs[i] /= norm

    # Measure: cosine similarities should be ~0 (no interference)
    if n_gs > 1:
        off_diag_sims = []
        for i in range(min(100, n_gs)):
            for j in range(i+1, min(100, n_gs)):
                off_diag_sims.append(abs(float(np.dot(gs_embs[i], gs_embs[j]))))
        mean_interference = float(np.mean(off_diag_sims)) if off_diag_sims else 0.0
    else:
        mean_interference = 0.0

    # Measure: semantic retrieval accuracy (should be ~0)
    orig_sims = embeddings[:n_gs] @ embeddings[:n_gs].T
    gs_sims = gs_embs @ gs_embs.T

    # For each item, check if nearest neighbor in GS space matches original
    gs_correct = 0
    for i in range(min(200, n_gs)):
        orig_nn = np.argsort(orig_sims[i])[-2]  # skip self
        gs_nn = np.argsort(gs_sims[i])[-2]
        if orig_nn == gs_nn:
            gs_correct += 1
    gs_accuracy = gs_correct / min(200, n_gs)

    results['gram_schmidt'] = {
        'mean_interference': mean_interference,
        'semantic_accuracy': gs_accuracy,
        'n_vectors': n_gs,
        'immunity': 1.0 - mean_interference,
        'usefulness': gs_accuracy,
    }

    # Method 2: Random projection
    for proj_dim in [32, 64, 128, 256]:
        proj_matrix = np.random.randn(dim, proj_dim) / np.sqrt(proj_dim)
        proj_embs = embeddings[:1000] @ proj_matrix
        proj_embs = proj_embs / (np.linalg.norm(proj_embs, axis=1, keepdims=True) + 1e-8)

        # Measure interference
        from noescape.utils import compute_participation_ratio
        pr = compute_participation_ratio(proj_embs)

        # Measure semantic accuracy
        orig_nn_sims = embeddings[:200] @ embeddings[:200].T
        proj_nn_sims = proj_embs[:200] @ proj_embs[:200].T
        rp_correct = 0
        for i in range(200):
            orig_nn = np.argsort(orig_nn_sims[i])[-2]
            proj_nn = np.argsort(proj_nn_sims[i])[-2]
            if orig_nn == proj_nn:
                rp_correct += 1

        results[f'random_proj_{proj_dim}'] = {
            'd_eff': float(pr['d_eff']),
            'semantic_accuracy': rp_correct / 200,
            'proj_dim': proj_dim,
        }

    return results


def solution4_compression(config: dict, encode_fn, wiki_sentences: list) -> Dict:
    """
    Solution 4: Memory compression via clustering and centroid merging.

    Test: reduces interference but degrades specific fact retrieval.
    """
    set_seed(42)
    from sklearn.cluster import MiniBatchKMeans

    texts = [s['text'] for s in wiki_sentences[:5000]]
    embeddings = encode_fn(texts)

    sol_cfg = config['solutions']['compression']
    merge_threshold = sol_cfg.get('merge_threshold', 0.9)

    results = {}
    noise_sigma = 0.25

    for n_clusters in [50, 100, 250, 500, 1000, 2500]:
        if n_clusters >= len(embeddings):
            continue

        # Cluster
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

        # Measure interference (mini Ebbinghaus)
        dim = centroids.shape[1]
        n_trials = 50
        age_bins = np.linspace(0.1, 30, 5)
        correct_counts = np.zeros(len(age_bins))
        total_counts = np.zeros(len(age_bins))

        for trial in range(n_trials):
            target_idx = trial % len(centroids)
            target = centroids[target_idx]

            for bi, age in enumerate(age_bins):
                noise_scale = noise_sigma * np.sqrt(age + 0.01) / np.sqrt(dim)
                noisy = target + noise_scale * np.random.randn(dim)
                noisy = noisy / (np.linalg.norm(noisy) + 1e-8)

                sims = centroids @ noisy
                correct_counts[bi] += int(np.argmax(sims) == target_idx)
                total_counts[bi] += 1

        accs = correct_counts / np.maximum(total_counts, 1)
        fit = fit_forgetting_curve(age_bins, accs)

        # Measure specific retrieval accuracy
        # For each original embedding, find nearest centroid, check if same cluster
        # Then retrieve: can we find the original?
        nearest_centroids = centroids[labels]
        retrieval_correct = 0
        for i in range(min(500, len(embeddings))):
            orig = embeddings[i]
            # Nearest centroid
            sims = centroids @ orig
            best_cluster = np.argmax(sims)
            # Is this the right cluster?
            if best_cluster == labels[i]:
                retrieval_correct += 1

        results[str(n_clusters)] = {
            'n_clusters': n_clusters,
            'compression_ratio': len(embeddings) / n_clusters,
            'fitted_b': fit['b'],
            'r_squared': fit['r_squared'],
            'cluster_retrieval_accuracy': retrieval_correct / min(500, len(embeddings)),
            'immunity': max(0, 1.0 - fit['b']),
            'usefulness': retrieval_correct / min(500, len(embeddings)),
        }

    return results


def run_all_solutions(config: dict, encode_fn, wiki_sentences: list) -> Dict:
    """Run all four solution analyses."""
    results = {}

    print("Solution 1: High nominal dimensionality...")
    results['high_dim'] = solution1_high_dimensionality(config, encode_fn, wiki_sentences)

    print("Solution 2: BM25 keyword retrieval...")
    results['bm25'] = solution2_bm25_keyword(config, wiki_sentences, encode_fn)

    print("Solution 3: Orthogonalization...")
    results['orthogonalization'] = solution3_orthogonalization(config, encode_fn, wiki_sentences)

    print("Solution 4: Memory compression...")
    results['compression'] = solution4_compression(config, encode_fn, wiki_sentences)

    return results


def save_results(results: dict, output_dir: str = 'results/solutions'):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'solution_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved solution analysis to {output_dir}/solution_analysis.json")
