"""
Experiment C: Spacing Effect

Tests whether distributed practice produces better retention than massed practice.

For embedding architectures: encode target fact with 3 repetitions at different
temporal spacings, add age-proportional noise, measure retrieval accuracy.

For LLM architectures: repeat target fact at different spacings in context window
with filler facts between repetitions.
"""

import numpy as np
import json
import os
from typing import Dict, List, Optional
from scipy.stats import wilcoxon

from noescape.utils import set_seed, bootstrap_confidence_interval, load_wikipedia_sentences


def run_embedding_spacing(architecture, config: dict, seed: int,
                           wiki_sentences: List[Dict] = None) -> dict:
    """
    Spacing experiment for embedding-based architectures.

    Protocol:
    - Target fact encoded 3 times at different temporal positions
    - 4 conditions: massed, short, medium, long spacing
    - Add distractors and noise
    - Measure retrieval at test_day=30
    """
    set_seed(seed)

    exp_cfg = config['experiments']['spacing']
    arch_cfg = config['architectures'].get(architecture.arch_key, {})
    n_facts = exp_cfg.get('n_facts', 100)
    n_distractors = exp_cfg.get('n_distractors', 25000)
    noise_sigma = arch_cfg.get('noise_sigma', exp_cfg.get('noise_sigma', 0.25))
    test_day = exp_cfg.get('test_day', 30)

    conditions_cfg = exp_cfg['conditions']

    if wiki_sentences is None:
        wiki_sentences = load_wikipedia_sentences(n_sentences=30000, n_articles=1000)

    # Separate target facts and distractors
    target_texts = [s['text'] for s in wiki_sentences[:n_facts]]
    distractor_texts = [s['text'] for s in wiki_sentences[n_facts:n_facts + n_distractors]]

    # Encode targets and distractors
    target_embs = architecture.encode(target_texts)
    if distractor_texts:
        distractor_embs = architecture.encode(distractor_texts[:5000])  # Limit for memory
    else:
        distractor_embs = np.zeros((0, target_embs.shape[1]))

    dim = target_embs.shape[1]
    results_per_condition = {}

    for cond_name, cond_cfg in conditions_cfg.items():
        gap_days = cond_cfg['gap_days']  # [presentation_1, presentation_2, presentation_3]
        correct = 0
        total = 0

        for i in range(min(n_facts, len(target_embs))):
            target_emb = target_embs[i]

            # Create 3 rehearsal embeddings at different times
            rehearsal_embs = []
            for t in gap_days:
                noise_scale = noise_sigma * np.sqrt(t + 0.01) / np.sqrt(dim)
                noisy = target_emb + noise_scale * np.random.randn(dim)
                noisy = noisy / (np.linalg.norm(noisy) + 1e-8)
                rehearsal_embs.append(noisy)

            # Average rehearsal embeddings (memory consolidation via repetition)
            consolidated = np.mean(rehearsal_embs, axis=0)
            consolidated = consolidated / (np.linalg.norm(consolidated) + 1e-8)

            # Apply noise at test time (age = test_day)
            test_noise_scale = noise_sigma * np.sqrt(test_day + 0.01) / np.sqrt(dim)
            test_emb = consolidated + test_noise_scale * np.random.randn(dim)
            test_emb = test_emb / (np.linalg.norm(test_emb) + 1e-8)

            # Build memory pool: test_emb + distractor subset
            n_dist = min(1000, distractor_embs.shape[0])
            if n_dist > 0:
                dist_subset = distractor_embs[np.random.choice(distractor_embs.shape[0], n_dist, replace=False)]
                # Apply noise to distractors
                dist_ages = np.random.uniform(0, test_day, n_dist)
                noisy_dists = []
                for d_emb, d_age in zip(dist_subset, dist_ages):
                    d_noise = noise_sigma * np.sqrt(d_age + 0.01) / np.sqrt(dim)
                    nd = d_emb + d_noise * np.random.randn(dim)
                    nd = nd / (np.linalg.norm(nd) + 1e-8)
                    noisy_dists.append(nd)
                all_embs = np.vstack([test_emb.reshape(1, -1), np.array(noisy_dists)])
            else:
                all_embs = test_emb.reshape(1, -1)

            # Query with original target
            query = target_emb / (np.linalg.norm(target_emb) + 1e-8)
            sims = all_embs @ query
            retrieved_idx = np.argmax(sims)

            correct += int(retrieved_idx == 0)
            total += 1

        retention = correct / max(total, 1)
        results_per_condition[cond_name] = {
            'retention': retention,
            'correct': correct,
            'total': total,
        }

    # Check ordering
    retentions = {k: v['retention'] for k, v in results_per_condition.items()}
    ordering_correct = (
        retentions.get('long', 0) > retentions.get('medium', 0) >=
        retentions.get('short', 0) >= retentions.get('massed', 0)
    )

    return {
        'seed': seed,
        'conditions': results_per_condition,
        'ordering_correct': ordering_correct,
    }


def run_llm_spacing(architecture, config: dict, seed: int,
                     wiki_sentences: List[Dict] = None) -> dict:
    """
    Spacing experiment for LLM-based architectures.

    Protocol: repeat target fact at different spacings in context with fillers.
    """
    set_seed(seed)

    exp_cfg = config['experiments']['spacing']
    n_facts = min(exp_cfg.get('n_facts', 100), 30)  # Reduced for LLM speed

    if wiki_sentences is None:
        wiki_sentences = load_wikipedia_sentences(n_sentences=5000, n_articles=200)

    filler_texts = [s['text'] for s in wiki_sentences]
    target_texts = filler_texts[:n_facts]
    filler_pool = filler_texts[n_facts:]

    spacing_configs = {
        'massed': 0,
        'short': 10,
        'medium': 50,
        'long': 200,
    }

    results_per_condition = {}

    for cond_name, gap_fillers in spacing_configs.items():
        correct = 0
        total = 0

        for i in range(min(n_facts, len(target_texts))):
            target = target_texts[i]
            question = f"What does the following fact state: {target[:50]}...?"

            # Build context: target repeated 3 times with fillers between
            context_facts = []
            fillers_used = 0

            # First presentation
            context_facts.append(target)

            # Fillers
            for j in range(gap_fillers):
                if fillers_used < len(filler_pool):
                    context_facts.append(filler_pool[fillers_used])
                    fillers_used += 1

            # Second presentation
            context_facts.append(target)

            # Fillers
            for j in range(gap_fillers):
                if fillers_used < len(filler_pool):
                    context_facts.append(filler_pool[fillers_used])
                    fillers_used += 1

            # Third presentation
            context_facts.append(target)

            # Additional fillers after (simulate passage of time)
            for j in range(min(100, len(filler_pool) - fillers_used)):
                context_facts.append(filler_pool[fillers_used])
                fillers_used += 1

            # Limit context length
            context_facts = context_facts[:500]

            if hasattr(architecture, 'answer_question_in_context'):
                answer = architecture.answer_question_in_context(context_facts, question)
                target_words = set(target.lower().split())
                answer_words = set(answer.lower().split())
                overlap = len(target_words & answer_words) / max(len(target_words), 1)
                is_correct = overlap > 0.3
            else:
                architecture.clear()
                architecture.store(context_facts)
                results = architecture.retrieve(question, top_k=1)
                if results:
                    retrieved_text = architecture.get_item_text(results[0][0])
                    is_correct = (retrieved_text == target)
                else:
                    is_correct = False

            correct += int(is_correct)
            total += 1

        retention = correct / max(total, 1)
        results_per_condition[cond_name] = {
            'retention': retention,
            'correct': correct,
            'total': total,
        }

    retentions = {k: v['retention'] for k, v in results_per_condition.items()}
    ordering_correct = (
        retentions.get('long', 0) > retentions.get('medium', 0) >=
        retentions.get('short', 0) >= retentions.get('massed', 0)
    )

    return {
        'seed': seed,
        'conditions': results_per_condition,
        'ordering_correct': ordering_correct,
    }


def run_experiment(architecture, config: dict, seed: int,
                   wiki_sentences: List[Dict] = None) -> dict:
    if architecture.arch_key in ('vector_db', 'graph'):
        return run_embedding_spacing(architecture, config, seed, wiki_sentences)
    else:
        return run_llm_spacing(architecture, config, seed, wiki_sentences)


def run_all_seeds(architecture, config: dict, seeds: list = None,
                  wiki_sentences: List[Dict] = None) -> dict:
    if seeds is None:
        seeds = config['seeds']

    per_seed = {}
    for seed in seeds:
        print(f"Running Spacing seed={seed} for {architecture.name}...")
        result = run_experiment(architecture, config, seed, wiki_sentences)
        per_seed[str(seed)] = result

    # Aggregate
    conditions = ['massed', 'short', 'medium', 'long']
    aggregated = {}
    for cond in conditions:
        values = [per_seed[str(s)]['conditions'][cond]['retention'] for s in seeds
                  if cond in per_seed[str(s)]['conditions']]
        arr = np.array(values) if values else np.array([0.0])
        ci = bootstrap_confidence_interval(arr) if len(arr) > 1 else (arr[0], arr[0])
        aggregated[f'{cond}_mean'] = float(np.mean(arr))
        aggregated[f'{cond}_std'] = float(np.std(arr))
        aggregated[f'{cond}_ci'] = list(ci)

    # Cohen's d for long vs massed
    long_vals = np.array([per_seed[str(s)]['conditions']['long']['retention'] for s in seeds
                          if 'long' in per_seed[str(s)]['conditions']])
    massed_vals = np.array([per_seed[str(s)]['conditions']['massed']['retention'] for s in seeds
                            if 'massed' in per_seed[str(s)]['conditions']])

    if len(long_vals) > 1 and len(massed_vals) > 1:
        pooled_std = np.sqrt(((len(long_vals)-1)*np.var(long_vals, ddof=1) +
                              (len(massed_vals)-1)*np.var(massed_vals, ddof=1)) /
                             (len(long_vals) + len(massed_vals) - 2))
        cohens_d = float((np.mean(long_vals) - np.mean(massed_vals)) / max(pooled_std, 1e-8))
    else:
        cohens_d = 0.0

    try:
        stat, p = wilcoxon(long_vals, massed_vals, alternative='greater')
        aggregated['wilcoxon_p_long_vs_massed'] = float(p)
    except Exception:
        aggregated['wilcoxon_p_long_vs_massed'] = 1.0

    aggregated['cohens_d_long_vs_massed'] = cohens_d
    aggregated['ordering_correct_count'] = sum(
        per_seed[str(s)]['ordering_correct'] for s in seeds
    )

    return {
        'architecture': architecture.arch_key,
        'experiment': 'spacing',
        'per_seed': per_seed,
        'aggregated': aggregated,
    }


def save_results(results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'spacing.json')
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved Spacing results to {path}")
