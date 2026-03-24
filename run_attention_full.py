#!/usr/bin/env python3
"""
Architecture 2: Full Attention Memory experiments.
This will take ~5-7 hours. Run as background task.

Runs: Ebbinghaus (full), Spacing, TOT
(DRM already done)
"""

import sys, os, json, time, yaml, torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from noescape.utils import set_seed, load_wikipedia_sentences, load_drm_word_lists, bootstrap_confidence_interval, fit_forgetting_curve


def main():
    t_start = time.time()
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    seeds = config['seeds']
    wiki = load_wikipedia_sentences(n_sentences=5000, n_articles=200)
    print(f"Data: {len(wiki)} sentences")

    # Group by article
    articles = {}
    for s in wiki:
        aid = s['article_id']
        if aid not in articles:
            articles[aid] = []
        articles[aid].append(s['text'])
    article_ids = [a for a in articles if len(articles[a]) >= 2]

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = config['models']['qwen']['hf_id']
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True)
    except Exception:
        fallback = config['models']['qwen'].get('fallback', 'Qwen/Qwen2.5-3B-Instruct')
        print(f"Falling back to {fallback}")
        tokenizer = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            fallback, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True)
    model.eval()
    print("Model loaded.")

    def generate(prompt, max_tokens=50):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    results_dir = Path('results/attention')
    results_dir.mkdir(parents=True, exist_ok=True)

    # ========== EBBINGHAUS (full protocol from spec 3.2) ==========
    print("\n=== EBBINGHAUS (full protocol) ===")
    n_near_values = [0, 10, 50, 100, 200, 500, 1000]
    n_target_facts = 50  # Reduced from 100 for feasibility (still 50×5×7×5 = 8750 calls)
    n_positions = 5

    ebb_per_seed = {}
    for seed in seeds:
        set_seed(seed)
        np.random.shuffle(article_ids)
        print(f"\n  Seed {seed}...")

        results_per_n_near = {}
        for n_near in n_near_values:
            t0 = time.time()
            print(f"    n_near={n_near}...", end=" ", flush=True)

            # Collect target facts and competitors
            per_position_correct = np.zeros(n_positions)
            per_position_total = np.zeros(n_positions)

            n_trials = min(n_target_facts, len(article_ids))
            for trial in range(n_trials):
                target_aid = article_ids[trial % len(article_ids)]
                target_sents = articles[target_aid]
                target_text = target_sents[0]

                # Competitors: from same and other articles
                competitors = list(target_sents[1:])
                for aid in article_ids:
                    if aid != target_aid:
                        competitors.extend(articles[aid][:3])
                    if len(competitors) >= n_near:
                        break
                competitors = competitors[:n_near]

                total_facts = 1 + len(competitors)
                question = f"What does this fact state: {target_text[:60]}...?"

                # Test at 5 positions
                for pos_idx in range(n_positions):
                    if total_facts <= 1:
                        target_pos = 0
                    else:
                        frac = pos_idx / (n_positions - 1)
                        target_pos = int(frac * (total_facts - 1))

                    fact_list = list(competitors)
                    fact_list.insert(min(target_pos, len(fact_list)), target_text)
                    fact_list = fact_list[:500]  # Context length limit

                    facts_text = "\n".join([f"{i+1}. {f}" for i, f in enumerate(fact_list)])
                    prompt = (f"You are a factual assistant. Answer based ONLY on facts below.\n\n"
                             f"FACTS:\n{facts_text}\n\nQUESTION: {question}\nAnswer:")

                    answer = generate(prompt)
                    target_words = set(target_text.lower().split()) - {'the','a','an','is','was','of','in','to','and','that','it','for'}
                    answer_words = set(answer.lower().split()) - {'the','a','an','is','was','of','in','to','and','that','it','for'}
                    overlap = len(target_words & answer_words) / max(len(target_words), 1)
                    correct = overlap > 0.3

                    per_position_correct[pos_idx] += int(correct)
                    per_position_total[pos_idx] += 1

            # Age = position-based (position 0 = oldest, position N = newest)
            simulated_days = 30
            valid = per_position_total > 0
            ages = np.array([(n_positions - i) / n_positions * simulated_days for i in range(n_positions)])[valid]
            accuracies = (per_position_correct / np.maximum(per_position_total, 1))[valid]

            fit = fit_forgetting_curve(ages, accuracies)
            results_per_n_near[str(n_near)] = {
                'ages': ages.tolist(), 'accuracies': accuracies.tolist(),
                'fitted_b': fit['b'], 'fitted_a': fit['a'], 'fitted_c': fit.get('c', 0),
                'r_squared': fit['r_squared'], 'fit_success': fit['fit_success'],
            }
            elapsed = time.time() - t0
            print(f"b={fit['b']:.3f} R²={fit['r_squared']:.3f} ({elapsed:.0f}s)")

        ebb_per_seed[str(seed)] = {
            'seed': seed, 'n_near_values': n_near_values,
            'per_n_near': results_per_n_near,
        }

        # Save after each seed
        first = ebb_per_seed[str(seeds[0])]
        ebb_agg = {'per_n_near': {}}
        for nk in first['per_n_near']:
            bvals = [ebb_per_seed[str(s)]['per_n_near'][nk]['fitted_b'] for s in seeds if str(s) in ebb_per_seed]
            arr = np.array(bvals)
            ci = bootstrap_confidence_interval(arr) if len(arr) > 1 else (arr[0], arr[0])
            ebb_agg['per_n_near'][nk] = {
                'b_mean': float(np.mean(arr)), 'b_std': float(np.std(arr)),
                'b_ci_lower': ci[0], 'b_ci_upper': ci[1], 'n_seeds': len(bvals),
            }
        ebb_results = {'architecture': 'attention', 'experiment': 'ebbinghaus',
                       'per_seed': ebb_per_seed, 'aggregated': ebb_agg}
        with open(results_dir / 'ebbinghaus.json', 'w') as f:
            json.dump(ebb_results, f, indent=2, default=str)
        print(f"  Saved after seed {seed}")

    # ========== SPACING ==========
    print("\n=== SPACING ===")
    spacing_configs = {'massed': 0, 'short': 10, 'medium': 50, 'long': 200}
    n_spacing_facts = 30  # Reduced for feasibility

    sp_per_seed = {}
    for seed in seeds:
        set_seed(seed)
        print(f"  Seed {seed}...")
        cond_results = {}

        for cond_name, gap_fillers in spacing_configs.items():
            correct = 0
            total = 0

            filler_pool = [s['text'] for s in wiki[100:]]
            target_texts = [s['text'] for s in wiki[:n_spacing_facts]]

            for i, target in enumerate(target_texts):
                question = f"What does this fact state: {target[:60]}...?"

                # Build context: target repeated 3x with fillers
                context = []
                fillers_used = 0
                for rep in range(3):
                    context.append(target)
                    for _ in range(gap_fillers):
                        if fillers_used < len(filler_pool):
                            context.append(filler_pool[fillers_used])
                            fillers_used += 1
                # Add trailing fillers
                for _ in range(min(100, len(filler_pool) - fillers_used)):
                    context.append(filler_pool[fillers_used])
                    fillers_used += 1

                context = context[:500]
                facts_text = "\n".join([f"{j+1}. {f}" for j, f in enumerate(context)])
                prompt = (f"Answer based ONLY on facts below.\n\nFACTS:\n{facts_text}\n\n"
                         f"QUESTION: {question}\nAnswer:")

                answer = generate(prompt)
                target_words = set(target.lower().split()) - {'the','a','an','is','was','of','in','to','and'}
                answer_words = set(answer.lower().split()) - {'the','a','an','is','was','of','in','to','and'}
                overlap = len(target_words & answer_words) / max(len(target_words), 1)
                correct += int(overlap > 0.3)
                total += 1

            retention = correct / max(total, 1)
            cond_results[cond_name] = {'retention': float(retention), 'correct': correct, 'total': total}
            print(f"    {cond_name}: {retention:.3f}")

        rets = [cond_results[c]['retention'] for c in ['massed', 'short', 'medium', 'long']]
        ordering = all(rets[i] <= rets[i+1] for i in range(3))
        sp_per_seed[str(seed)] = {'seed': seed, 'conditions': cond_results, 'ordering_correct': ordering}

    # Aggregate spacing
    sp_agg = {}
    for c in ['massed', 'short', 'medium', 'long']:
        vals = [sp_per_seed[str(s)]['conditions'][c]['retention'] for s in seeds]
        arr = np.array(vals)
        ci = bootstrap_confidence_interval(arr) if len(arr) > 1 else (arr[0], arr[0])
        sp_agg[f'{c}_mean'] = float(np.mean(arr))
        sp_agg[f'{c}_std'] = float(np.std(arr))
        sp_agg[f'{c}_ci'] = list(ci)
    lv = np.array([sp_per_seed[str(s)]['conditions']['long']['retention'] for s in seeds])
    mv = np.array([sp_per_seed[str(s)]['conditions']['massed']['retention'] for s in seeds])
    ps = np.sqrt(((len(lv)-1)*np.var(lv,ddof=1)+(len(mv)-1)*np.var(mv,ddof=1))/(len(lv)+len(mv)-2))
    sp_agg['cohens_d_long_vs_massed'] = float((np.mean(lv)-np.mean(mv))/max(ps, 1e-8))
    sp_agg['ordering_correct_count'] = sum(sp_per_seed[str(s)]['ordering_correct'] for s in seeds)

    sp_results = {'architecture': 'attention', 'experiment': 'spacing',
                  'per_seed': sp_per_seed, 'aggregated': sp_agg}
    with open(results_dir / 'spacing.json', 'w') as f:
        json.dump(sp_results, f, indent=2, default=str)
    print(f"  Spacing saved. long={sp_agg['long_mean']:.3f} > massed={sp_agg['massed_mean']:.3f}: {sp_agg['long_mean'] > sp_agg['massed_mean']}")

    # ========== TOT ==========
    print("\n=== TOT ===")
    n_tot_facts = 100  # Present 100 facts, query each

    tot_per_seed = {}
    for seed in seeds:
        set_seed(seed)
        print(f"  Seed {seed}...", end=" ", flush=True)

        facts = [s['text'] for s in wiki[:500]]
        context_facts = facts[:100]
        n_tot = 0
        n_correct = 0
        n_total = 0

        for i, fact in enumerate(context_facts):
            # Present surrounding facts as context
            start = max(0, i - 25)
            end = min(len(facts), i + 25)
            context = facts[start:end]

            question = f"What is the specific content of this fact: {fact[:50]}...?"
            facts_text = "\n".join([f"{j+1}. {f}" for j, f in enumerate(context)])
            prompt = (f"Answer based on the facts.\n\nFACTS:\n{facts_text}\n\n"
                     f"QUESTION: {question}\nAnswer:")

            answer = generate(prompt, max_tokens=30)

            fact_words = set(fact.lower().split()) - {'the','a','an','is','was','of','in','to','and','that','it','for'}
            answer_words = set(answer.lower().split()) - {'the','a','an','is','was','of','in','to','and','that','it','for'}
            if not fact_words:
                continue

            n_total += 1
            overlap = len(fact_words & answer_words) / len(fact_words)

            if overlap > 0.5:
                n_correct += 1
            elif overlap > 0.1:
                n_tot += 1  # TOT: partial match

        tot_rate = n_tot / max(n_total, 1)
        print(f"TOT={tot_rate:.4f} ({n_tot}/{n_total})")
        tot_per_seed[str(seed)] = {'seed': seed, 'tot_rate': float(tot_rate),
                                    'n_tot_states': n_tot, 'n_queries': n_total, 'n_correct': n_correct}

    rates = np.array([tot_per_seed[str(s)]['tot_rate'] for s in seeds])
    ci = bootstrap_confidence_interval(rates) if len(rates) > 1 else (rates[0], rates[0])
    tot_results = {'architecture': 'attention', 'experiment': 'tot',
                   'per_seed': tot_per_seed,
                   'aggregated': {'tot_rate_mean': float(np.mean(rates)), 'tot_rate_std': float(np.std(rates)),
                                  'tot_rate_ci': list(ci)}}
    with open(results_dir / 'tot.json', 'w') as f:
        json.dump(tot_results, f, indent=2, default=str)
    print(f"  TOT saved. Mean={np.mean(rates):.4f}")

    print(f"\n=== Architecture 2 COMPLETE. Total time: {(time.time()-t_start)/60:.1f} min ===")


if __name__ == '__main__':
    main()
