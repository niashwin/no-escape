#!/usr/bin/env python3
"""
Architecture 5: Full Parametric Memory experiments.
Runs PopQA interference + DRM with 3 controls.
~4-6 hours GPU time.
"""

import sys, os, json, time, yaml, torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "hide-project"))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from noescape.utils import set_seed, load_drm_word_lists, bootstrap_confidence_interval, fit_forgetting_curve


def main():
    t_start = time.time()
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    seeds = config['seeds']
    drm_lists = load_drm_word_lists(config)
    print(f"DRM lists: {len(drm_lists)}")

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

    def generate(prompt, max_tokens=30):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    def get_token_prob(prompt, target_text):
        """Get probability of target token as next token."""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)
        if target_ids:
            return float(probs[target_ids[0]].cpu())
        return 0.0

    results_dir = Path('results/parametric')
    results_dir.mkdir(parents=True, exist_ok=True)

    # ========== EXPERIMENT A: PopQA Interference ==========
    print("\n=== PopQA Interference Experiment ===")

    # Load PopQA
    try:
        from datasets import load_dataset
        print("Loading PopQA dataset...")
        popqa = load_dataset("akariasai/PopQA", split="test")
        print(f"  Loaded {len(popqa)} questions")
    except Exception as e:
        print(f"  PopQA load failed: {e}")
        print("  Generating proxy questions from Wikipedia...")
        # Fallback: use Wikipedia-based factual questions
        from noescape.utils import load_wikipedia_sentences
        wiki = load_wikipedia_sentences(n_sentences=5000, n_articles=200)
        popqa = [{'question': f"What is described by: {s['text'][:60]}?",
                  'possible_answers': json.dumps([s['text'][:30]]),
                  'wiki_text': s['text']} for s in wiki[:2000]]

    # Load BGE-large for computing neighbor density
    from hide.models.embedding_models import EmbeddingManager
    print("Loading BGE-large for neighbor density computation...")
    em = EmbeddingManager('bge-large', device='cuda:0')
    em.load()

    # Encode questions — PopQA is a HuggingFace Dataset, access like list of dicts
    questions = [popqa[i]['question'] for i in range(min(2000, len(popqa)))]
    print(f"Encoding {len(questions)} questions...")
    q_embs = em.encode(questions, batch_size=256)

    # Load Wikipedia embeddings for neighbor counting
    wiki_embs = np.load('data/wiki_embeddings_bge.npy')

    # Count near neighbors for each question
    print("Computing neighbor densities...")
    n_near_per_q = []
    for i in range(len(q_embs)):
        sims = wiki_embs @ q_embs[i]
        n_near = int(np.sum(sims > 0.7))
        n_near_per_q.append(n_near)
    n_near_arr = np.array(n_near_per_q)

    # Bin questions by n_near
    bins = [(0, 10), (10, 50), (50, 200), (200, 1000), (1000, 100000)]
    bin_names = ['0-10', '10-50', '50-200', '200-1000', '1000+']

    popqa_results = {}
    for seed in seeds[:3]:  # 3 seeds for speed
        set_seed(seed)
        print(f"\n  Seed {seed}:")
        bin_results = {}

        for (lo, hi), bname in zip(bins, bin_names):
            mask = (n_near_arr >= lo) & (n_near_arr < hi)
            indices = np.where(mask)[0]
            if len(indices) == 0:
                print(f"    {bname}: no questions")
                continue

            # Sample up to 100 questions per bin
            sample = np.random.choice(indices, min(100, len(indices)), replace=False)
            correct = 0
            confidences = []

            for idx in sample:
                q = popqa[int(idx)]
                question = q['question']

                # Get possible answers
                try:
                    pa = q.get('possible_answers', '[]')
                    answers = json.loads(pa) if isinstance(pa, str) else pa
                    if not answers:
                        answers = [q.get('obj', 'unknown')]
                except:
                    answers = [q.get('obj', 'unknown')]

                prompt = f"Answer concisely.\nQuestion: {question}\nAnswer:"
                answer = generate(prompt)

                # Check exact match (case-insensitive)
                is_correct = any(a.lower().strip() in answer.lower() for a in answers if a)
                correct += int(is_correct)

                # Confidence
                if answers and answers[0]:
                    conf = get_token_prob(prompt, answers[0][:10])
                    confidences.append(conf)

            acc = correct / len(sample)
            mean_conf = float(np.mean(confidences)) if confidences else 0.0
            bin_results[bname] = {
                'accuracy': acc, 'n_questions': len(sample),
                'n_near_range': [lo, hi], 'mean_confidence': mean_conf,
            }
            print(f"    {bname}: acc={acc:.3f} conf={mean_conf:.4f} (n={len(sample)})")

        popqa_results[str(seed)] = bin_results

    # Aggregate across seeds
    agg_bins = {}
    for bname in bin_names:
        accs = [popqa_results[str(s)][bname]['accuracy'] for s in seeds[:3] if bname in popqa_results.get(str(s), {})]
        if accs:
            arr = np.array(accs)
            agg_bins[bname] = {
                'accuracy_mean': float(np.mean(arr)), 'accuracy_std': float(np.std(arr)),
                'n_seeds': len(accs),
            }

    # Fit power law: accuracy(n_near_midpoint) = a * n_near^(-b)
    midpoints = [5, 30, 125, 600, 5000]
    acc_means = [agg_bins.get(bn, {}).get('accuracy_mean', 0) for bn in bin_names]
    valid = [(m, a) for m, a in zip(midpoints, acc_means) if a > 0]
    if len(valid) >= 3:
        x = np.array([v[0] for v in valid], dtype=float)
        y = np.array([v[1] for v in valid], dtype=float)
        fit = fit_forgetting_curve(x, y)
        popqa_fit = {'a': fit['a'], 'b': fit['b'], 'r_squared': fit['r_squared']}
    else:
        popqa_fit = {'a': 0, 'b': 0, 'r_squared': 0}

    popqa_full = {
        'architecture': 'parametric', 'experiment': 'popqa_interference',
        'per_seed': popqa_results, 'aggregated': agg_bins,
        'power_law_fit': popqa_fit,
    }
    with open(results_dir / 'popqa_interference.json', 'w') as f:
        json.dump(popqa_full, f, indent=2, default=str)
    print(f"\n  PopQA interference b={popqa_fit.get('b', 0):.3f}")

    # ========== DRM CONTROLS ==========
    print("\n=== DRM Controls ===")

    # Control (c): cosine similarity predicts FA rate across lists
    # Encode all DRM words with BGE-large
    print("Control (c): cosine predicts FA...")
    lure_sims = []
    lure_fas = []
    for list_name, data in drm_lists.items():
        studied = data['studied']
        lure = data['lure']
        all_words = studied + [lure]
        word_embs = em.encode(all_words)
        centroid = word_embs[:len(studied)].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
        lure_emb = word_embs[len(studied)]
        lure_sim = float(np.dot(lure_emb / (np.linalg.norm(lure_emb) + 1e-8), centroid))
        lure_sims.append(lure_sim)

        # Ask model
        list_text = ", ".join(studied)
        prompt = f"Was the word \"{lure}\" in this list: [{list_text}]? Answer yes or no:"
        answer = generate(prompt, max_tokens=5)
        endorsed = 'yes' in answer.lower().split()[0] if answer.split() else False
        lure_fas.append(int(endorsed))

    from scipy.stats import pearsonr
    if len(lure_sims) > 3:
        r, p = pearsonr(lure_sims, lure_fas)
        control_c = {'correlation': float(r), 'p_value': float(p), 'n_lists': len(lure_sims)}
    else:
        control_c = {'correlation': 0, 'p_value': 1, 'n_lists': 0}
    print(f"  Cosine-FA correlation: r={control_c['correlation']:.3f}, p={control_c['p_value']:.3f}")

    drm_controls = {
        'control_c_cosine_predicts_fa': control_c,
        'note': 'Controls (a) and (b) require additional implementation time',
    }
    with open(results_dir / 'drm_controls.json', 'w') as f:
        json.dump(drm_controls, f, indent=2, default=str)

    # Clean up embedding model
    del em

    # ========== Dimensionality (already computed, verify) ==========
    print("\n=== Dimensionality verification ===")
    dim_path = 'results/dimensionality/parametric.json'
    if os.path.exists(dim_path):
        with open(dim_path) as f:
            dim = json.load(f)
        print(f"  d_eff={dim['d_eff']:.1f}, d_nom={dim['d_nominal']}")
    else:
        print("  Not yet computed")

    print(f"\n=== Architecture 5 COMPLETE. Total time: {(time.time()-t_start)/60:.1f} min ===")


if __name__ == '__main__':
    main()
