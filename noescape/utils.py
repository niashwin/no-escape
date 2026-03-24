"""
Shared utilities for the No Escape experiments.

Includes: bootstrap CI, power-law fitting, data loading, seeding,
statistical tests, and results I/O.

Wherever possible, reuse the metrics and utilities from the original
HIDE codebase (hide-project/hide/utils/metrics.py).
"""

import json
import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Add HIDE project to path for reuse
HIDE_ROOT = Path(__file__).parent.parent / "hide-project"
sys.path.insert(0, str(HIDE_ROOT))

# Reuse HIDE metrics where possible
try:
    from hide.utils.metrics import (
        bootstrap_ci,
        fit_power_law,
        participation_ratio,
        cohens_d,
        r_squared,
    )
    HIDE_METRICS_AVAILABLE = True
except ImportError:
    HIDE_METRICS_AVAILABLE = False


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def bootstrap_confidence_interval(data: np.ndarray, n_resamples: int = 10000,
                                   confidence: float = 0.95,
                                   statistic=np.mean) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of values.
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        statistic: Function to compute statistic (default: np.mean).

    Returns:
        (lower, upper) bounds of the CI.
    """
    if len(data) == 0:
        return (np.nan, np.nan)

    boot_stats = np.array([
        statistic(np.random.choice(data, size=len(data), replace=True))
        for _ in range(n_resamples)
    ])

    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


def fit_forgetting_curve(ages: np.ndarray, accuracies: np.ndarray) -> Dict:
    """
    Fit power-law forgetting curve: accuracy(age) = a * age^(-b) + c

    Returns dict with fitted parameters, R², and CI for b.
    """
    from scipy.optimize import curve_fit

    def power_law(t, a, b, c):
        return a * np.power(t + 1e-6, -b) + c

    try:
        popt, pcov = curve_fit(
            power_law, ages, accuracies,
            p0=[1.0, 0.5, 0.0],
            bounds=([0, 0, 0], [1, 5, 0.5]),
            maxfev=10000
        )
        a, b, c = popt
        predicted = power_law(ages, *popt)
        ss_res = np.sum((accuracies - predicted) ** 2)
        ss_tot = np.sum((accuracies - np.mean(accuracies)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {
            'a': float(a),
            'b': float(b),
            'c': float(c),
            'r_squared': float(r2),
            'fit_success': True,
        }
    except Exception as e:
        return {
            'a': 0, 'b': 0, 'c': 0, 'r_squared': 0,
            'fit_success': False, 'error': str(e),
        }


def compute_participation_ratio(embeddings: np.ndarray) -> Dict:
    """
    Compute effective dimensionality via participation ratio.

    d_eff = (Σλ_i)² / Σ(λ_i²)

    Also returns d_95 and d_99 (components for 95%/99% variance).
    """
    # Center
    centered = embeddings - embeddings.mean(axis=0)
    # Covariance
    cov = np.cov(centered.T)
    # Eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # descending
    eigenvalues = np.maximum(eigenvalues, 0)  # numerical stability

    # Participation ratio
    sum_lambda = eigenvalues.sum()
    sum_lambda_sq = (eigenvalues ** 2).sum()
    d_eff = (sum_lambda ** 2) / sum_lambda_sq if sum_lambda_sq > 0 else 0

    # Cumulative explained variance
    total_var = eigenvalues.sum()
    if total_var > 0:
        cumvar = np.cumsum(eigenvalues) / total_var
        d_95 = int(np.searchsorted(cumvar, 0.95) + 1)
        d_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    else:
        d_95 = d_99 = 0

    return {
        'd_eff': float(d_eff),
        'd_95': int(d_95),
        'd_99': int(d_99),
        'd_nominal': embeddings.shape[1],
        'eigenvalues_top100': eigenvalues[:100].tolist(),
        'explained_variance_ratio': (eigenvalues / total_var)[:100].tolist() if total_var > 0 else [],
    }


def levina_bickel_estimator(embeddings: np.ndarray, k1: int = 1, k2: int = 2) -> float:
    """
    Levina-Bickel two-nearest-neighbor intrinsic dimensionality estimator.

    Reference: Levina & Bickel (2005), "Maximum Likelihood Estimation of
    Intrinsic Dimension"
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k2 + 1).fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    # distances[:, 0] is self (0), distances[:, k1] is k1-th neighbor, etc.
    d_k1 = distances[:, k1]
    d_k2 = distances[:, k2]

    # Filter out zero distances
    mask = (d_k1 > 0) & (d_k2 > 0)
    d_k1 = d_k1[mask]
    d_k2 = d_k2[mask]

    # MLE estimator
    log_ratios = np.log(d_k2 / d_k1)
    d_hat = 1.0 / np.mean(log_ratios) if np.mean(log_ratios) > 0 else np.inf

    return float(d_hat)


def aggregate_seed_results(seed_results: List[Dict], key: str) -> Dict:
    """
    Aggregate a metric across seeds with bootstrap CI.

    Args:
        seed_results: List of result dicts (one per seed).
        key: Key to extract from each dict.

    Returns:
        Dict with mean, std, ci_lower, ci_upper, values.
    """
    values = np.array([r[key] for r in seed_results if key in r and r[key] is not None])
    if len(values) == 0:
        return {'mean': None, 'std': None, 'ci_lower': None, 'ci_upper': None, 'values': []}

    ci = bootstrap_confidence_interval(values)
    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'ci_lower': ci[0],
        'ci_upper': ci[1],
        'values': values.tolist(),
    }


def load_drm_word_lists(config: dict) -> Dict:
    """Load DRM word lists from the HIDE project data.

    Returns dict mapping list name -> {'studied': [...], 'lure': '...'}
    Falls back to embedded DRM_LISTS from HIDE emergent.py if file has different structure.
    """
    drm_path = Path(__file__).parent.parent / config['datasets']['drm']['path']
    if not drm_path.exists():
        drm_path = HIDE_ROOT / "data" / "drm_word_lists.json"

    try:
        with open(drm_path) as f:
            data = json.load(f)
        # Handle nested structure: {'source': ..., 'license': ..., 'lists': {...}}
        if 'lists' in data and isinstance(data['lists'], dict):
            return data['lists']
        # Check if top-level keys are list names
        first_key = next(iter(data))
        if isinstance(data[first_key], dict) and 'studied' in data[first_key]:
            return data
    except Exception:
        pass

    # Fallback to HIDE embedded lists
    try:
        from hide.core.emergent import DRM_LISTS
        return DRM_LISTS
    except ImportError:
        pass

    # Final fallback: return a minimal set
    return {
        "SLEEP": {"studied": ["bed","rest","awake","tired","dream","wake","snooze","blanket","doze","slumber","snore","nap","peace","yawn","drowsy"], "lure": "sleep"},
        "NEEDLE": {"studied": ["thread","pin","eye","sewing","sharp","point","prick","thimble","haystack","thorn","hurt","injection","syringe","cloth","knitting"], "lure": "needle"},
    }


def load_wikipedia_sentences(n_sentences: int = 10000, n_articles: int = 500,
                             sentences_per_article: int = 20) -> List[Dict]:
    """
    Load Wikipedia sentences for experiments.

    Returns list of dicts with 'text', 'article_id', 'article_title'.
    """
    cache_path = Path(__file__).parent.parent / "data" / "wiki_sentences_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if len(cached) >= n_sentences:
            return cached[:n_sentences]

    sentences = []

    # Try loading Wikipedia with a timeout-safe approach
    try:
        from datasets import load_dataset
        print("  Attempting to load Wikipedia from HuggingFace (streaming)...")
        ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                        split="train", streaming=True)
        article_count = 0
        for article in ds:
            if article_count >= n_articles:
                break
            text = article.get('text', '')
            sents = [s.strip() for s in text.split('.') if len(s.strip()) > 30][:sentences_per_article]
            for s in sents:
                sentences.append({
                    'text': s + '.',
                    'article_id': article_count,
                    'article_title': article.get('title', f'article_{article_count}'),
                })
            article_count += 1
            if len(sentences) >= n_sentences:
                break
            if article_count % 100 == 0:
                print(f"    Loaded {article_count} articles, {len(sentences)} sentences...")
    except Exception as e:
        print(f"  Wikipedia download failed: {e}")
        print("  Using generated factual corpus instead.")

    if len(sentences) < n_sentences:
        sentences = _generate_factual_sentences(n_sentences, sentences)

    sentences = sentences[:n_sentences]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(sentences, f)

    return sentences


def _generate_factual_sentences(n_target: int, existing: List[Dict]) -> List[Dict]:
    """Generate diverse factual sentences for experiments when Wikipedia unavailable."""
    import random

    # Factual knowledge domains for diversity
    topics = {
        'geography': [
            "The Amazon River is approximately 6,400 kilometers long and flows through South America",
            "Mount Everest stands at 8,849 meters above sea level in the Himalayas",
            "The Sahara Desert covers approximately 9.2 million square kilometers in Africa",
            "Lake Baikal in Russia contains about 20 percent of the world's surface fresh water",
            "The Nile River flows northward through northeastern Africa for about 6,650 kilometers",
            "The Pacific Ocean covers approximately 165.25 million square kilometers",
            "The Great Barrier Reef stretches for over 2,300 kilometers along the Australian coast",
            "The Dead Sea is approximately 430 meters below sea level",
            "Iceland sits on the Mid-Atlantic Ridge where two tectonic plates meet",
            "The Grand Canyon is approximately 446 kilometers long and up to 29 kilometers wide",
        ],
        'science': [
            "The speed of light in a vacuum is approximately 299,792 kilometers per second",
            "Water molecules consist of two hydrogen atoms and one oxygen atom",
            "The human body contains approximately 206 bones in the adult skeleton",
            "DNA has a double helix structure first described by Watson and Crick in 1953",
            "The Earth's core is primarily composed of iron and nickel at temperatures above 5,000 degrees Celsius",
            "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight",
            "The periodic table currently contains 118 confirmed chemical elements",
            "Neurons transmit electrical signals at speeds up to 120 meters per second",
            "The human genome contains approximately 3 billion base pairs of DNA",
            "Gravity on Earth accelerates objects at approximately 9.8 meters per second squared",
        ],
        'history': [
            "The ancient Egyptian pyramids at Giza were built approximately 4,500 years ago",
            "The Roman Empire at its greatest extent covered approximately 5 million square kilometers",
            "The printing press was invented by Johannes Gutenberg around 1440",
            "The French Revolution began in 1789 with the storming of the Bastille",
            "The Industrial Revolution started in Britain in the late 18th century",
            "Alexander the Great created one of the largest empires in ancient history by age 30",
            "The Great Wall of China was built over many centuries starting from the 7th century BCE",
            "The Renaissance period in Europe spanned roughly from the 14th to the 17th century",
            "The first successful powered airplane flight by the Wright Brothers occurred in 1903",
            "The Berlin Wall fell on November 9, 1989, reunifying East and West Germany",
        ],
        'biology': [
            "The blue whale is the largest animal ever known to have lived on Earth",
            "Trees communicate through underground fungal networks called mycorrhizal networks",
            "The human brain contains approximately 86 billion neurons",
            "Photosynthetic organisms produce most of the oxygen in Earth's atmosphere",
            "The cheetah can reach speeds of up to 112 kilometers per hour",
            "Octopuses have three hearts and blue blood due to copper-based hemocyanin",
            "The average adult human body contains about 37.2 trillion cells",
            "Honeybees communicate the location of food sources through waggle dances",
            "The human eye can distinguish approximately 10 million different colors",
            "Tardigrades can survive extreme conditions including the vacuum of outer space",
        ],
        'technology': [
            "The first electronic general-purpose computer ENIAC was completed in 1945",
            "The internet originated from ARPANET which was developed in the late 1960s",
            "Moore's Law predicted that transistor density would double approximately every two years",
            "The World Wide Web was invented by Tim Berners-Lee in 1989",
            "Silicon semiconductors form the basis of modern computer processors",
            "The first smartphone was the IBM Simon, released in 1994",
            "Artificial neural networks were inspired by biological neural networks in the brain",
            "The GPS satellite navigation system requires at least four satellites for accurate positioning",
            "Quantum computers use qubits that can exist in superposition of states",
            "The first successful test of email was conducted by Ray Tomlinson in 1971",
        ],
        'astronomy': [
            "The Sun is approximately 150 million kilometers from Earth",
            "Jupiter is the largest planet in our solar system with a mass 318 times that of Earth",
            "The Milky Way galaxy contains an estimated 100 to 400 billion stars",
            "Light from the nearest star Proxima Centauri takes about 4.24 years to reach Earth",
            "The observable universe has a diameter of approximately 93 billion light-years",
            "Saturn's rings are primarily composed of ice particles and rocky debris",
            "A neutron star can have a mass greater than the Sun compressed into a sphere 20 kilometers across",
            "The cosmic microwave background radiation was discovered accidentally in 1965",
            "Mars has the tallest known mountain in the solar system, Olympus Mons at 21.9 kilometers",
            "The Andromeda Galaxy is approximately 2.5 million light-years from the Milky Way",
        ],
        'culture': [
            "The Mona Lisa was painted by Leonardo da Vinci in the early 16th century",
            "Shakespeare wrote approximately 37 plays and 154 sonnets during his lifetime",
            "The ancient Olympic Games were held in Olympia, Greece beginning in 776 BCE",
            "Beethoven composed his Ninth Symphony while he was almost completely deaf",
            "The Rosetta Stone was discovered in 1799 and helped decode Egyptian hieroglyphics",
            "Chess originated in India during the Gupta Empire around the 6th century CE",
            "The Library of Alexandria was one of the largest and most significant libraries of the ancient world",
            "Mozart composed his first symphony at the age of eight",
            "The Parthenon in Athens was completed in 438 BCE as a temple dedicated to Athena",
            "The first feature-length animated film was El Apostol, created in Argentina in 1917",
        ],
        'physics': [
            "Einstein's special theory of relativity established that E equals mc squared",
            "The standard model of particle physics describes 17 fundamental particles",
            "Superconductors have zero electrical resistance below a critical temperature",
            "The Higgs boson was experimentally confirmed at CERN in 2012",
            "Black holes have gravitational fields so strong that nothing can escape past the event horizon",
            "The wave-particle duality suggests that all matter exhibits both wave and particle properties",
            "Absolute zero is 0 Kelvin, equivalent to minus 273.15 degrees Celsius",
            "The strong nuclear force is the strongest of the four fundamental forces of nature",
            "Heisenberg's uncertainty principle states you cannot simultaneously know exact position and momentum",
            "The theory of general relativity predicts that massive objects warp the fabric of spacetime",
        ],
    }

    sentences = list(existing)
    all_facts = []
    for topic_name, facts in topics.items():
        for i, fact in enumerate(facts):
            all_facts.append({
                'text': fact + '.',
                'article_id': len(set(s.get('article_id', -1) for s in sentences)) + hash(topic_name) % 1000,
                'article_title': f'{topic_name}_{i}',
            })

    # Shuffle and extend with variations
    random.seed(42)
    base_facts = all_facts.copy()

    while len(sentences) + len(all_facts) < n_target:
        for fact in base_facts:
            # Create variations by adding context
            prefixes = [
                "According to scientific research, ",
                "Historical records indicate that ",
                "It is well established that ",
                "Research has shown that ",
                "Studies have confirmed that ",
                "It is widely known that ",
                "Evidence suggests that ",
                "Experts have determined that ",
            ]
            prefix = random.choice(prefixes)
            new_text = prefix + fact['text'][0].lower() + fact['text'][1:]
            all_facts.append({
                'text': new_text,
                'article_id': fact['article_id'] + len(all_facts),
                'article_title': fact['article_title'] + '_var',
            })
            if len(sentences) + len(all_facts) >= n_target:
                break

    random.shuffle(all_facts)
    sentences.extend(all_facts)
    return sentences[:n_target]
