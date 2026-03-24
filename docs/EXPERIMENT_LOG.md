# Experiment Log

## Overview

This document records the experimental progression, key decisions, and lessons learned during the execution of the "No Escape" follow-up paper.

---

## Phase 1: Calibration Against HIDE

**Objective:** Reproduce HIDE's published results (b=0.460, DRM FA=0.583, spacing long>massed, TOT≈3.66%) using BGE-large embeddings.

**Challenge:** HIDE's published results used MiniLM (384-dim) for most experiments, but the spec calls for BGE-large (1024-dim) as the primary embedding model. This created a parameter mismatch — BGE-large has different similarity distributions than MiniLM.

**Key Decision: Decay parameter calibration.** The temporal decay function S(t) = (1 + βt)^(-ψ) with ψ=0.5 required tuning β for BGE-large. A sweep over β ∈ [0.01, 0.5] found **β=0.20** gives b=0.440 at 10K competitors — within one SE of HIDE's b=0.460.

**Key Decision: DRM threshold calibration.** HIDE used θ=0.82 (calibrated for MiniLM similarities). For BGE-large, which has systematically higher cosine similarities, the equivalent threshold is **θ=0.864**, where FA=0.583 matches exactly.

**Result:** All four calibration checks passed after parameter tuning:
- Ebbinghaus b = 0.440 ± 0.030 at 10K ✓
- DRM FA = 0.583 at θ=0.864 ✓
- Spacing long (0.902) > massed (0.360) ✓
- TOT = 0.020 (within tolerance) ✓

---

## Phase 2: Embedding Architecture Experiments (Vector DB + Graph)

**Ebbinghaus:** Both architectures show clear monotonic increase in forgetting exponent with competitor count. Vector DB reaches b=0.440, Graph reaches b=0.478 — both in the human range (0.3-0.7).

**DRM:** 24/24 lures fall within predicted spherical cap intersections. The geometric prediction of Theorem 3 is confirmed without exception.

**Spacing:** Vector DB shows strong long > massed effect (Cohen's d = 24.6). Graph shows the same direction with smaller magnitude due to lower interference at 384 dimensions.

**TOT:** PCA reduction to 96 dimensions + query noise σ=1.5/√96 produces TOT rates of 2.0% (VDB) and 2.8% (Graph) — close to HIDE's 3.66%.

---

## Phase 3: LLM Architecture Experiments (Attention + Parametric)

**Key Finding: Phase transition in attention-based memory.** The attention architecture (Qwen2.5-7B) does NOT show smooth power-law forgetting. Instead, it shows a phase transition: near-perfect accuracy with <100 competitors, then catastrophic failure at 200+. This is qualitatively different from embedding architectures but still demonstrates interference — the system has a capacity threshold beyond which interference is total.

**Key Finding: Parametric interference via PopQA.** The most novel experiment: using 14,267 PopQA questions binned by neighbour density in the training corpus. Accuracy drops monotonically from 1.000 (<50 neighbours) to 0.113 (>1000 neighbours). This is interference IN THE WEIGHTS — no external memory system involved.

**Key Finding: LLM DRM FA = 0 at behavioural level.** Both attention and parametric architectures correctly reject all DRM lures — the model can do exact string matching on word lists. This does NOT refute the theorem (which applies to representation geometry, not behavioural output) but is an important nuance the paper addresses.

**Key Finding: Parametric TOT = 69%.** Extremely high partial retrieval rate — the model frequently gives answers that are in the right domain but wrong on specifics. This is the tip-of-tongue phenomenon expressed in LLM generation.

---

## Phase 4: Filesystem Architecture (BM25 + LLM)

**Key Finding: BM25 escapes interference by escaping SPP.** Keyword matching produces b=0.000 (no forgetting), FA=0.000 (no false recall), and all spacing conditions at 1.000. But SPP correlation is only r=0.210 and semantic retrieval agreement is 15.5%. This architecture demonstrates Solution 2 (exact-match retrieval) from a different angle — it's simultaneously an architecture test and a solution analysis.

---

## Phase 5: Theorem Verification

**Anderson-Schooler:** At cosine threshold 0.5, we measure α = 0.459 (R² = 0.952) in the Wikipedia corpus — close to the expected α ≈ 0.5 from Anderson & Schooler (1991). This confirms that competitors arrive at a power-law rate in natural text.

**Spherical caps:** Analytical cap volumes match simulation within 20% for all tested (d, θ) combinations.

**DRM caps:** All 24/24 lures fall within the spherical cap intersection of their studied associates.

**d_eff convergence:** Participation ratio gives d_eff = 158 (BGE), 127 (MiniLM), 17.9 (Qwen). Levina-Bickel gives d_eff ≈ 10-15. Both confirm d_eff ≪ d_nom.

---

## Phase 6: Solution Analysis

All four solutions tested with recalibrated parameters (β=0.20):

1. **High dimensionality:** 7 data points. Zero-padding from 1024→4096 doesn't reduce b (d_eff unchanged at 124). PCA reduction does change b but reduces semantic accuracy.
2. **BM25 keyword retrieval:** Immunity = 1.0, Usefulness = 15.5%. Clear tradeoff.
3. **Orthogonalisation:** Gram-Schmidt gives immunity = 1.0 but accuracy = 0.0%. Random projection shows graded tradeoff.
4. **Compression:** 6 data points from 50 to 2500 clusters. Monotonic tradeoff between b and retrieval accuracy.

**No solution achieves both immunity (b < 0.1) and usefulness (accuracy > 70%).**

---

## Key Lessons Learned

1. **The two-level distinction is the real finding.** The original expectation was "all 5 architectures show the same 4 phenomena." Reality is more interesting: the geometric vulnerability is universal, but the behavioural expression depends on whether the system can build workarounds — and those workarounds are never free.

2. **LLMs are better at explicit tasks than expected.** DRM FA = 0 for all LLM architectures because the model can parse word lists. The theorem still holds at the representation level, but this means the paper must carefully distinguish geometric from behavioural predictions.

3. **Calibration is model-dependent.** BGE-large and MiniLM have different similarity distributions, requiring different thresholds and decay parameters. This is honest and documented, not a failure.

4. **The PopQA result is the strongest new finding.** Showing interference in model weights — accuracy declining monotonically with neighbour density — is something HIDE could not demonstrate because it only tested external memory systems.

---

## Compute Time

| Experiment | Time |
|:---|:---|
| Calibration (beta sweep + 5 seeds) | ~1 hour |
| Graph memory (all experiments) | ~30 min |
| Attention full (Ebbinghaus 5 seeds + Spacing + TOT) | ~4 hours |
| Parametric PopQA | ~30 min |
| Filesystem (BM25 + LLM DRM + SPP) | ~30 min |
| Remaining experiments | ~20 min |
| Solution analysis | ~30 min |
| Theorem verification | ~30 min |
| **Total** | **~8 hours GPU** |
