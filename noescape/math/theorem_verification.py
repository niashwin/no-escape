"""
Theorem Verification Module

Numerically verifies the three theorems and lemma from the No Escape paper:
- Theorem 1: Low effective dimensionality (rate-distortion)
- Theorem 2A: Spherical cap volumes match simulated interference
- Theorem 2B: Anderson-Schooler statistics in Wikipedia
- Theorem 3: DRM spherical cap intersection geometry
"""

import numpy as np
import json
import os
from typing import Dict, List
from scipy.special import betainc, gamma
from scipy.stats import ttest_rel
from pathlib import Path

from noescape.utils import (
    set_seed, compute_participation_ratio, levina_bickel_estimator,
    bootstrap_confidence_interval
)


def verify_spp_all_architectures(architectures: list, config: dict,
                                  related_pairs: List, unrelated_pairs: List) -> Dict:
    """
    Verify SPP for all 5 architectures.
    Paired t-test, p < 0.001 required.
    """
    results = {}
    for arch in architectures:
        print(f"  Verifying SPP for {arch.name}...")
        spp = arch.verify_spp(related_pairs, unrelated_pairs)
        results[arch.arch_key] = spp
        status = "PASS" if spp['spp_satisfied'] else "FAIL"
        print(f"    {status}: related={spp['related_mean']:.4f}, "
              f"unrelated={spp['unrelated_mean']:.4f}, p={spp['p_value']:.2e}")
    return results


def spherical_cap_volume(d: int, theta_deg: float) -> float:
    """
    Compute volume fraction of spherical cap on S^(d-1).

    Vol(Cap(theta)) / Vol(S^(d-1)) = (1/2) * I_{sin^2(theta)}((d-1)/2, 1/2)

    where I is the regularized incomplete beta function.
    """
    theta = np.radians(theta_deg)
    x = np.sin(theta) ** 2
    a = (d - 1) / 2.0
    b = 0.5
    return 0.5 * betainc(a, b, x)


def verify_spherical_caps(config: dict) -> Dict:
    """
    Verify Theorem 2A: spherical cap volumes match simulated interference.

    For different d_eff and theta values:
    1. Compute analytical cap volume
    2. Simulate random points on S^(d-1), count fraction in cap
    3. Compare analytical vs simulation
    """
    set_seed(42)
    thm_cfg = config['theorems']['spherical_cap']
    d_values = thm_cfg.get('d_eff_values', [8, 16, 32, 64, 128, 256])
    theta_values = thm_cfg.get('theta_degrees', [10, 20, 30, 45, 60])
    n_simulations = thm_cfg.get('n_simulation_trials', 100000)

    results = {}
    for d in d_values:
        results[str(d)] = {}
        for theta_deg in theta_values:
            # Analytical
            analytical_vol = spherical_cap_volume(d, theta_deg)

            # Simulation: random unit vectors, check angular distance to pole
            random_points = np.random.randn(n_simulations, d)
            random_points = random_points / np.linalg.norm(random_points, axis=1, keepdims=True)

            pole = np.zeros(d)
            pole[0] = 1.0

            cos_angles = random_points @ pole
            angles_deg = np.degrees(np.arccos(np.clip(cos_angles, -1, 1)))
            simulated_vol = float(np.mean(angles_deg < theta_deg))

            # Compare
            ratio = simulated_vol / max(analytical_vol, 1e-15)

            results[str(d)][str(theta_deg)] = {
                'analytical': float(analytical_vol),
                'simulated': simulated_vol,
                'ratio': ratio,
                'within_20pct': abs(ratio - 1.0) < 0.20 if analytical_vol > 1e-10 else True,
            }

    # Compute P_fail for interference
    p_fail_results = {}
    for d in [16, 32, 64]:
        p_near = spherical_cap_volume(d, 30)
        p_fail_by_n = {}
        for n_near in [10, 100, 1000, 10000]:
            p_fail = 1 - (1 - p_near) ** n_near
            p_fail_by_n[str(n_near)] = float(p_fail)
        p_fail_results[str(d)] = {
            'p_near': float(p_near),
            'p_fail': p_fail_by_n,
        }

    return {
        'cap_volumes': results,
        'interference_probability': p_fail_results,
    }


def verify_anderson_schooler(config: dict, wiki_sentences: list,
                              embeddings: np.ndarray = None) -> Dict:
    """
    Verify Theorem 2B: Anderson-Schooler statistics in Wikipedia.

    Measure inter-arrival time distribution of semantically near items.
    Fit power law, report alpha (expected ≈ 0.5).
    """
    set_seed(42)
    thm_cfg = config['theorems']['anderson_schooler']
    cosine_threshold = thm_cfg.get('cosine_threshold', 0.7)
    n_sentences = min(thm_cfg.get('n_sentences', 100000), len(wiki_sentences))

    texts = [s['text'] for s in wiki_sentences[:n_sentences]]

    if embeddings is None:
        # Need to encode - use the architecture's encoder
        import sys
        from pathlib import Path
        HIDE_ROOT = Path(__file__).parent.parent.parent / "hide-project"
        sys.path.insert(0, str(HIDE_ROOT))
        from hide.models.embedding_models import EmbeddingManager
        em = EmbeddingManager("bge-large", device="cuda:0")
        em.load()
        embeddings = em.encode(texts, batch_size=256, show_progress=True)

    # For each sentence, find inter-arrival time of semantically near items
    inter_arrival_times = []

    # Sample pairs efficiently
    n_sample = min(5000, len(embeddings))
    sample_idx = np.random.choice(len(embeddings), n_sample, replace=False)

    for idx in sample_idx:
        query = embeddings[idx]
        sims = embeddings @ query
        near_mask = sims > cosine_threshold
        near_mask[idx] = False  # exclude self
        near_indices = np.where(near_mask)[0]

        if len(near_indices) > 1:
            # Inter-arrival times = gaps between near items in sequence order
            sorted_near = np.sort(near_indices)
            gaps = np.diff(sorted_near)
            inter_arrival_times.extend(gaps.tolist())

    if not inter_arrival_times:
        return {'alpha': 0, 'fit_success': False, 'error': 'No near items found'}

    iat = np.array(inter_arrival_times, dtype=float)
    iat = iat[iat > 0]

    # Fit power law to inter-arrival time distribution
    # P(gap > t) ~ t^(-alpha)
    from scipy.optimize import curve_fit

    # Compute empirical survival function
    sorted_iat = np.sort(iat)
    survival = 1 - np.arange(1, len(sorted_iat) + 1) / len(sorted_iat)

    # Subsample for fitting
    n_fit = min(1000, len(sorted_iat))
    fit_idx = np.linspace(0, len(sorted_iat) - 1, n_fit, dtype=int)
    x_fit = sorted_iat[fit_idx]
    y_fit = survival[fit_idx]

    # Filter valid
    valid = (x_fit > 0) & (y_fit > 0)
    x_fit = x_fit[valid]
    y_fit = y_fit[valid]

    try:
        def power_law(t, a, alpha):
            return a * np.power(t, -alpha)

        popt, _ = curve_fit(power_law, x_fit, y_fit, p0=[1.0, 0.5],
                           bounds=([0, 0], [100, 3]), maxfev=10000)
        alpha = popt[1]

        y_pred = power_law(x_fit, *popt)
        ss_res = np.sum((y_fit - y_pred) ** 2)
        ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return {
            'alpha': float(alpha),
            'alpha_a': float(popt[0]),
            'r_squared': float(r2),
            'n_inter_arrivals': len(iat),
            'mean_gap': float(np.mean(iat)),
            'median_gap': float(np.median(iat)),
            'fit_success': True,
        }
    except Exception as e:
        return {'alpha': 0, 'fit_success': False, 'error': str(e)}


def verify_drm_caps(config: dict, drm_lists: dict = None,
                     encode_fn=None) -> Dict:
    """
    Verify Theorem 3: DRM lures lie within spherical cap intersections.

    For each DRM list:
    - Compute angular distances between all studied words and the lure
    - Verify lure falls within the cap intersection of studied words
    """
    set_seed(42)

    if drm_lists is None:
        from noescape.utils import load_drm_word_lists
        drm_lists = load_drm_word_lists(config)

    results = {}
    lure_in_cap_count = 0
    total_lists = 0

    for list_name, data in drm_lists.items():
        studied = data['studied']
        lure = data['lure']

        # Encode
        all_words = studied + [lure]
        embeddings = encode_fn(all_words)

        studied_embs = embeddings[:len(studied)]
        lure_emb = embeddings[len(studied)]

        # Normalize
        studied_embs = studied_embs / (np.linalg.norm(studied_embs, axis=1, keepdims=True) + 1e-8)
        lure_emb = lure_emb / (np.linalg.norm(lure_emb) + 1e-8)

        # Angular distances from lure to each studied word
        cos_sims = studied_embs @ lure_emb
        angular_distances = np.degrees(np.arccos(np.clip(cos_sims, -1, 1)))

        # Centroid
        centroid = studied_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        # Angular distances from centroid to studied words
        cos_studied_to_centroid = studied_embs @ centroid
        max_angular_from_centroid = float(np.degrees(np.arccos(
            np.clip(min(cos_studied_to_centroid), -1, 1))))

        # Angular distance from centroid to lure
        cos_lure_to_centroid = float(np.dot(lure_emb, centroid))
        angular_lure_to_centroid = float(np.degrees(np.arccos(
            np.clip(cos_lure_to_centroid, -1, 1))))

        # Lure is in cap if its angular distance to centroid < max studied angular distance
        in_cap = angular_lure_to_centroid < max_angular_from_centroid
        lure_in_cap_count += int(in_cap)
        total_lists += 1

        results[list_name] = {
            'angular_distances_to_lure': angular_distances.tolist(),
            'mean_angular_to_lure': float(np.mean(angular_distances)),
            'centroid_to_lure_angle': angular_lure_to_centroid,
            'max_studied_to_centroid_angle': max_angular_from_centroid,
            'lure_in_cap': in_cap,
            'lure_cosine_to_centroid': cos_lure_to_centroid,
        }

    return {
        'per_list': results,
        'lure_in_cap_rate': lure_in_cap_count / max(total_lists, 1),
        'n_in_cap': lure_in_cap_count,
        'n_total': total_lists,
    }


def verify_dimensionality_theorem(config: dict, embeddings: np.ndarray) -> Dict:
    """
    Verify Theorem 1: participation ratio ≈ Levina-Bickel estimator.
    Both should give d_eff ≈ 16 for BGE-large.
    """
    pr_result = compute_participation_ratio(embeddings)
    lb_result = levina_bickel_estimator(embeddings)

    d_eff_pr = pr_result['d_eff']
    d_eff_lb = lb_result

    ratio = d_eff_pr / max(d_eff_lb, 1e-8)
    agree = 0.5 < ratio < 2.0  # Within factor of 2

    return {
        'participation_ratio': d_eff_pr,
        'levina_bickel': d_eff_lb,
        'ratio_pr_over_lb': float(ratio),
        'agree_within_factor_2': agree,
        'd_95': pr_result['d_95'],
        'd_99': pr_result['d_99'],
        'd_nominal': pr_result['d_nominal'],
    }


def run_all_verifications(config: dict, wiki_sentences: list = None,
                           embeddings: np.ndarray = None,
                           encode_fn=None, drm_lists: dict = None) -> Dict:
    """Run all theorem verifications and save results."""
    results = {}

    print("Verifying spherical cap volumes (Theorem 2A)...")
    results['spherical_caps'] = verify_spherical_caps(config)

    if wiki_sentences and embeddings is not None:
        print("Verifying Anderson-Schooler statistics (Theorem 2B)...")
        results['anderson_schooler'] = verify_anderson_schooler(
            config, wiki_sentences, embeddings)

    if embeddings is not None:
        print("Verifying dimensionality theorem (Theorem 1)...")
        results['dimensionality'] = verify_dimensionality_theorem(config, embeddings)

    if encode_fn and drm_lists:
        print("Verifying DRM cap intersections (Theorem 3)...")
        results['drm_caps'] = verify_drm_caps(config, drm_lists, encode_fn)

    return results


def save_results(results: dict, output_dir: str = 'results/theorems'):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'theorem_verification.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved theorem verification to {output_dir}/theorem_verification.json")
