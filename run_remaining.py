#!/usr/bin/env python3
"""
Run remaining experiments:
- Arch 3 (Filesystem): Ebbinghaus via BM25+LLM, Spacing, TOT
- Arch 5 (Parametric): TOT (using same Qwen model)
Both use Qwen, so run sequentially.
"""

import sys, os, json, time, yaml, torch
import numpy as np
from pathlib import Path
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hide-project"))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from noescape.utils import set_seed, load_wikipedia_sentences, bootstrap_confidence_interval, fit_forgetting_curve


def main():
    t_start = time.time()
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    seeds = config['seeds']
    wiki = load_wikipedia_sentences(n_sentences=5000, n_articles=200)
    texts = [s['text'] for s in wiki]
    print(f"Data: {len(wiki)} sentences")

    # Load Qwen
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
        tokenizer = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            fallback, torch_dtype=torch.float16,
            device_map="auto", trust_remote_code=True)
    model.eval()
    print("Model loaded.")

    def generate(prompt, max_tokens=30):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    def llm_relevance(query, memory):
        prompt = (f"Rate relevance 1-10.\nQuery: {query}\nMemory: {memory}\nScore:")
        answer = generate(prompt, max_tokens=5)
        try:
            return max(1, min(10, int(''.join(c for c in answer if c.isdigit())[:2]))) / 10.0
        except:
            return 0.5

    # ================================================================
    # ARCHITECTURE 3: Filesystem - Ebbinghaus via BM25+LLM
    # ================================================================
    print("\n" + "="*60)
    print("Architecture 3: Filesystem Ebbinghaus")
    print("="*60)

    fs_dir = Path('results/filesystem')
    fs_dir.mkdir(parents=True, exist_ok=True)

    # Build BM25 index
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    n_near_values = [0, 50, 200, 500]  # Compute-limited per spec
    n_trials = 30  # Per n_near

    ebb_per_seed = {}
    for seed in seeds:
        set_seed(seed)
        print(f"\n  Seed {seed}:")

        articles = {}
        for s in wiki:
            aid = s['article_id']
            if aid not in articles:
                articles[aid] = []
            articles[aid].append(s['text'])
        article_ids = [a for a in articles if len(articles[a]) >= 2]
        np.random.shuffle(article_ids)

        results_per_n_near = {}
        for n_near in n_near_values:
            t0 = time.time()
            temporal_bins = 5
            per_bin_correct = np.zeros(temporal_bins)
            per_bin_total = np.zeros(temporal_bins)

            for trial in range(n_trials):
                target_aid = article_ids[trial % len(article_ids)]
                target_text = articles[target_aid][0]

                # Competitors
                competitors = list(articles[target_aid][1:])
                for aid in article_ids:
                    if aid != target_aid:
                        competitors.extend(articles[aid][:3])
                    if len(competitors) >= n_near:
                        break
                competitors = competitors[:n_near]

                # Build local BM25 index over competitors + target
                local_corpus = competitors + [target_text]
                local_tokenized = [t.lower().split() for t in local_corpus]
                local_bm25 = BM25Okapi(local_tokenized) if local_corpus else None

                total_facts = len(local_corpus)
                question = f"What does this state: {target_text[:50]}...?"

                for bin_idx in range(temporal_bins):
                    # Query via BM25 → top candidates → LLM rerank
                    q_tokens = question.lower().split()
                    if local_bm25:
                        scores = local_bm25.get_scores(q_tokens)
                        top5_idx = np.argsort(scores)[::-1][:5]

                        # LLM rerank top 5
                        best_idx = top5_idx[0]
                        best_rel = 0
                        for idx in top5_idx[:3]:  # Rerank top 3 for speed
                            if scores[idx] > 0:
                                rel = llm_relevance(question, local_corpus[idx])
                                if rel > best_rel:
                                    best_rel = rel
                                    best_idx = idx

                        correct = (best_idx == len(local_corpus) - 1)  # Target is last
                    else:
                        correct = True  # No competitors

                    per_bin_correct[bin_idx] += int(correct)
                    per_bin_total[bin_idx] += 1

            valid = per_bin_total > 0
            ages = np.array([(i+1)/temporal_bins * 30 for i in range(temporal_bins)])[valid]
            accuracies = (per_bin_correct / np.maximum(per_bin_total, 1))[valid]
            fit = fit_forgetting_curve(ages, accuracies)

            results_per_n_near[str(n_near)] = {
                'ages': ages.tolist(), 'accuracies': accuracies.tolist(),
                'fitted_b': fit['b'], 'fitted_a': fit['a'],
                'r_squared': fit['r_squared'], 'fit_success': fit['fit_success'],
            }
            elapsed = time.time() - t0
            print(f"    n_near={n_near}: b={fit['b']:.3f} ({elapsed:.0f}s)")

        ebb_per_seed[str(seed)] = {
            'seed': seed, 'n_near_values': n_near_values,
            'per_n_near': results_per_n_near,
        }

    # Aggregate
    first = ebb_per_seed[str(seeds[0])]
    ebb_agg = {'per_n_near': {}}
    for nk in first['per_n_near']:
        bvals = [ebb_per_seed[str(s)]['per_n_near'][nk]['fitted_b'] for s in seeds]
        arr = np.array(bvals)
        ci = bootstrap_confidence_interval(arr) if len(arr) > 1 else (arr[0], arr[0])
        ebb_agg['per_n_near'][nk] = {
            'b_mean': float(np.mean(arr)), 'b_std': float(np.std(arr)),
            'b_ci_lower': ci[0], 'b_ci_upper': ci[1], 'n_seeds': len(bvals),
        }
    ebb_r = {'architecture': 'filesystem', 'experiment': 'ebbinghaus',
             'per_seed': ebb_per_seed, 'aggregated': ebb_agg}
    with open(fs_dir / 'ebbinghaus.json', 'w') as f:
        json.dump(ebb_r, f, indent=2, default=str)

    # ================================================================
    # ARCHITECTURE 3: Filesystem - TOT
    # ================================================================
    print("\n=== Filesystem TOT ===")
    tot_per_seed = {}
    for seed in seeds:
        set_seed(seed)
        facts = texts[:100]
        n_tot = 0; n_correct = 0; n_total = 0

        for i, fact in enumerate(facts):
            context = texts[max(0,i-25):min(len(texts),i+25)]
            question = f"What is: {fact[:50]}...?"

            # BM25 retrieval
            q_tokens = question.lower().split()
            scores = bm25.get_scores(q_tokens)
            top_idx = np.argmax(scores)
            retrieved = texts[top_idx]

            # Check if it's the right fact
            fact_words = set(fact.lower().split()) - {'the','a','an','is','was','of','in','to','and','that','it','for'}
            ret_words = set(retrieved.lower().split()) - {'the','a','an','is','was','of','in','to','and','that','it','for'}
            if not fact_words: continue
            n_total += 1
            overlap = len(fact_words & ret_words) / len(fact_words)
            if overlap > 0.5:
                n_correct += 1
            elif overlap > 0.1:
                n_tot += 1

        rate = n_tot / max(n_total, 1)
        tot_per_seed[str(seed)] = {'seed': seed, 'tot_rate': rate, 'n_tot_states': n_tot, 'n_queries': n_total}
        print(f"  seed={seed}: TOT={rate:.4f}")

    rates = np.array([tot_per_seed[str(s)]['tot_rate'] for s in seeds])
    ci = bootstrap_confidence_interval(rates)
    with open(fs_dir / 'tot.json', 'w') as f:
        json.dump({'architecture': 'filesystem', 'experiment': 'tot', 'per_seed': tot_per_seed,
                   'aggregated': {'tot_rate_mean': float(np.mean(rates)), 'tot_rate_std': float(np.std(rates)),
                                  'tot_rate_ci': list(ci)}}, f, indent=2, default=str)

    # ================================================================
    # ARCHITECTURE 5: Parametric - TOT
    # ================================================================
    print("\n=== Parametric TOT ===")
    tot5_per_seed = {}
    for seed in seeds:
        set_seed(seed)
        facts = texts[:100]
        n_tot = 0; n_correct = 0; n_total = 0

        for i, fact in enumerate(facts):
            question = f"What is the content of: {fact[:50]}...?"
            answer = generate(f"Answer concisely.\nQuestion: {question}\nAnswer:")

            fact_words = set(fact.lower().split()) - {'the','a','an','is','was','of','in','to','and','that','it','for'}
            ans_words = set(answer.lower().split()) - {'the','a','an','is','was','of','in','to','and','that','it','for'}
            if not fact_words: continue
            n_total += 1
            overlap = len(fact_words & ans_words) / len(fact_words)
            if overlap > 0.5:
                n_correct += 1
            elif overlap > 0.1:
                n_tot += 1

        rate = n_tot / max(n_total, 1)
        tot5_per_seed[str(seed)] = {'seed': seed, 'tot_rate': rate, 'n_tot_states': n_tot, 'n_queries': n_total}
        print(f"  seed={seed}: TOT={rate:.4f}")

    rates5 = np.array([tot5_per_seed[str(s)]['tot_rate'] for s in seeds])
    ci5 = bootstrap_confidence_interval(rates5)
    param_dir = Path('results/parametric')
    with open(param_dir / 'tot.json', 'w') as f:
        json.dump({'architecture': 'parametric', 'experiment': 'tot', 'per_seed': tot5_per_seed,
                   'aggregated': {'tot_rate_mean': float(np.mean(rates5)), 'tot_rate_std': float(np.std(rates5)),
                                  'tot_rate_ci': list(ci5)}}, f, indent=2, default=str)

    # ================================================================
    # ARCHITECTURE 3: Filesystem - Spacing
    # ================================================================
    print("\n=== Filesystem Spacing ===")
    spacing_configs = {'massed': 0, 'short': 10, 'medium': 50, 'long': 200}
    n_sp_facts = 20

    sp_per_seed = {}
    for seed in seeds:
        set_seed(seed)
        cond_results = {}
        for cond_name, gap in spacing_configs.items():
            correct = 0; total = 0
            for i in range(n_sp_facts):
                target = texts[i]
                question = f"What is: {target[:50]}...?"

                # Build corpus: target repeated 3x with fillers
                corpus = []
                filler_idx = n_sp_facts
                for rep in range(3):
                    corpus.append(target)
                    for _ in range(gap):
                        if filler_idx < len(texts):
                            corpus.append(texts[filler_idx])
                            filler_idx += 1
                corpus = corpus[:200]

                # BM25 on this corpus
                local_tok = [t.lower().split() for t in corpus]
                local_bm25 = BM25Okapi(local_tok)
                scores = local_bm25.get_scores(question.lower().split())
                top_idx = np.argmax(scores)
                is_correct = corpus[top_idx] == target
                correct += int(is_correct)
                total += 1

            cond_results[cond_name] = {'retention': correct / max(total, 1)}

        rets = [cond_results[c]['retention'] for c in ['massed','short','medium','long']]
        ordering = all(rets[i] <= rets[i+1] for i in range(3))
        sp_per_seed[str(seed)] = {'seed': seed, 'conditions': cond_results, 'ordering_correct': ordering}
        print(f"  seed={seed}: M={rets[0]:.3f} S={rets[1]:.3f} Med={rets[2]:.3f} L={rets[3]:.3f}")

    sp_agg = {}
    for c in ['massed','short','medium','long']:
        vals = [sp_per_seed[str(s)]['conditions'][c]['retention'] for s in seeds]
        arr = np.array(vals)
        ci_ = bootstrap_confidence_interval(arr)
        sp_agg[f'{c}_mean'] = float(np.mean(arr))
        sp_agg[f'{c}_std'] = float(np.std(arr))
        sp_agg[f'{c}_ci'] = list(ci_)
    lv = np.array([sp_per_seed[str(s)]['conditions']['long']['retention'] for s in seeds])
    mv = np.array([sp_per_seed[str(s)]['conditions']['massed']['retention'] for s in seeds])
    ps = np.sqrt(((len(lv)-1)*np.var(lv,ddof=1)+(len(mv)-1)*np.var(mv,ddof=1))/(len(lv)+len(mv)-2))
    sp_agg['cohens_d_long_vs_massed'] = float((np.mean(lv)-np.mean(mv))/max(ps,1e-8))
    sp_agg['ordering_correct_count'] = sum(sp_per_seed[str(s)]['ordering_correct'] for s in seeds)

    with open(fs_dir / 'spacing.json', 'w') as f:
        json.dump({'architecture':'filesystem','experiment':'spacing',
                   'per_seed':sp_per_seed,'aggregated':sp_agg}, f, indent=2, default=str)

    print(f"\n=== ALL REMAINING EXPERIMENTS COMPLETE. Total: {(time.time()-t_start)/60:.1f} min ===")


if __name__ == '__main__':
    main()
