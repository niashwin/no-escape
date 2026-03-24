# The Price of Meaning: Why Every Semantic Memory System Forgets

**Organising memory by meaning makes forgetting and false recall inevitable. Scaling up does not fix it.**

Sambartha Ray Barman, Andrey Starenky, Sophia Bodnar, Nikhil Narasimhan, Ashwin Gopinath (Sentra.app / MIT)

## Overview

This repository contains the complete code, data, figures, and paper for the "No Escape" follow-up to the HIDE paper ("The Geometry of Forgetting"). We prove that any memory system satisfying the Semantic Proximity Property — semantically related items represented more similarly than unrelated items — must exhibit power-law forgetting, false recall, and partial retrieval states. We verify this across five architecturally distinct memory systems.

See [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) for the full motivation and [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) for the experimental record.

## Repository Structure

```
├── noescape/                    # Core library
│   ├── architectures/           # 5 memory architecture implementations
│   ├── experiments/             # Ebbinghaus, DRM, Spacing, TOT, dimensionality
│   ├── analysis/                # Figure generation + statistics
│   ├── math/                    # Theorem verification
│   ├── solutions/               # Solution analysis (4 proposed cures)
│   └── utils.py                 # Shared utilities
├── results/                     # Raw experimental results (36 JSON files)
│   ├── vector_db/               # Architecture 1
│   ├── graph/                   # Architecture 4
│   ├── attention/               # Architecture 2
│   ├── parametric/              # Architecture 5
│   ├── filesystem/              # Architecture 3
│   ├── dimensionality/          # d_eff per architecture
│   ├── theorems/                # Theorem verification
│   ├── solutions/               # Solution analysis
│   └── verification.json        # Success criteria
├── figures/                     # All figures (PDF + PNG)
├── paper/                       # LaTeX source + compiled PDF
├── docs/                        # Project context + experiment log
├── run_calibration_v2.py        # Architecture 1 calibration
├── run_attention_full.py        # Architecture 2 full experiments
├── run_parametric_full.py       # Architecture 5 PopQA experiments
├── run_filesystem_full.py       # Architecture 3 experiments
├── run_remaining.py             # Supplementary experiments
├── config.yaml                  # All hyperparameters
└── requirements.txt             # Python dependencies
```

## Quick Start

```bash
pip install -r requirements.txt

# Reproduce Architecture 1 (Vector DB) — ~1 hour
python run_calibration_v2.py

# Reproduce Architecture 2 (Attention) — ~4 hours
python run_attention_full.py

# Reproduce Architecture 5 (Parametric/PopQA) — ~30 min
python run_parametric_full.py

# Regenerate all figures from raw data
python -c "from noescape.analysis.figures import generate_all_figures; generate_all_figures('results', 'figures')"
```

## Requirements

- NVIDIA A100 80GB GPU (or equivalent)
- Python 3.11+
- ~10 GPU-hours for full reproduction

## Models Used (all open-weight)

- BAAI/bge-large-en-v1.5 (MIT) — 1024-dim sentence embeddings
- sentence-transformers/all-MiniLM-L6-v2 (Apache 2.0) — 384-dim sentence embeddings
- Qwen/Qwen2.5-7B-Instruct (Apache 2.0) — LLM for attention/parametric/filesystem

## Key Results

| Architecture | Forgetting b | DRM FA | d_eff |
|:---|:---|:---|:---|
| Vector DB | 0.440 ± 0.030 | 0.583 | 158 / 10.6 |
| Graph | 0.478 ± 0.028 | 0.208 | 127 |
| Attention | phase transition | 0.000† | 17.9 |
| Parametric | 0.215* | 0.000† | 17.9 |
| Filesystem | 0.000 | 0.000 | 158 |

\* PopQA interference b. † Behavioural; geometric prediction holds (24/24 caps).

## Citation

```bibtex
@article{barman2025priceofmeaning,
  author = {Barman, Sambartha Ray and Starenky, Andrey and Bodnar, Sophia and Narasimhan, Nikhil and Gopinath, Ashwin},
  title = {The Price of Meaning: Why Every Semantic Memory System Forgets},
  year = {2025}
}
```

## Acknowledgements

Computational experiments and manuscript preparation were assisted by Claude (Anthropic).
