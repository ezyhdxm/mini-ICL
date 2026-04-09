# Empirical Evidence Needed for the Revised Paper

This document maps the claims in the revised `mathematical_v2.tex` (Section 4) to the empirical evidence that currently exists or needs to be produced. It also covers downstream sections (Sec 5, 6) that reference the new framework.

---

## Section 4.1 — Task Retrieval: ANOVA Framework and P0-P3

### What the theory claims

- For memorized tasks, the ANOVA residual ratio SS_within/SS_total should vanish at long context.
- The cell means should decompose additively (small interaction).
- The interpolation coefficients should track the Bayesian posterior.

### What exists already

| Claim | Evidence | Status |
|-------|----------|--------|
| P0: low residual variance at long context | Table 1 in `task_vectors_v3.tex` (K=3, all memorized) | **Done** |
| P0: residual ratio across positions | `task_vector_r2_combined.png` (Appendix) | **Done** |
| P1: small interaction | Table 1(b) in `task_vectors_v3.tex` | **Done** |
| P2: interpolation R^2 | `averaging_r2_combined.png` | **Done** |
| P3: posterior alignment | `beta_alpha_traj_*_simplex.png` | **Done** |
| Causal intervention on task subspace | Table 2 in `ortho_intervention_v2.tex` | **Done** |

### What may need updating

1. **Scope clarification in Sec 5 text.** Table 1 is for K=3 models (all tasks memorized). The revised Sec 4.1 explicitly frames these as the "task retrieval regime." A sentence in `task_vectors_v3.tex` should note this: these results validate P0-P3 in the retrieval setting. *(Minor text edit, no new experiments.)*

---

## Section 4.2 — Generalization: ANOVA Breaks Down

### What the theory claims

1. **No meaningful task effect** for unmemorized tasks — the model has no stored theta_{z'}.
2. **In-context statistics are not f(z, s_t)** — hidden states depend on the full prefix, so within-cell variance doesn't vanish.
3. **No task vectors** for novel tasks.
4. **Representational decoupling** — the two modes occupy different subspaces.

### What exists already

| Claim | Evidence | Status |
|-------|----------|--------|
| ANOVA R^2 is lower for minor tasks | `natural_r2_major_vs_all_v6.png`, `natural_r2_major_vs_all_v8.png` | **Done** (currently only for E3/latent) |
| Interventional R^2 for minor tasks | `task_vector_r2_coin_k10_minor.png`, `task_vector_r2_latent_k10_minor.png`, `task_vector_r2_linear_k10_minor.png` | **Done** (all three tasks) |
| Natural-sequence R^2 for latent | `task_vector_r2_latent_k10_natural.png`, `task_vector_r2_latent_k10_v6_long_natural.png` | **Done** |
| OOD R^2 drops with diversity | `ood_r2_c4_l10_t4_logx_c16.png` (Fig 4 in paper) | **Done** |
| Phase transition M1->M2 | `kl_transition_combined_logx.png` (Fig 3 in paper) | **Done** |
| Orthogonal subspace intervention | Table 2 in paper | **Done** |
| Orthogonal subspace encodes context stats | Mentioned in `ortho_intervention_v2.tex`, detailed in Appendix | **Done** |

### What is MISSING or needs to be produced

#### Priority 1: Major-vs-minor ANOVA comparison (key evidence for Sec 4.2)

This is the central empirical evidence supporting the new "ANOVA doesn't properly apply" argument. We need a clean figure/table comparing:

- **Major-task R^2** (memorized, retrieval mode) vs. **Minor-task R^2** (unmemorized, generalization mode) — in the same model, at the same layers and positions.
- The gap quantifies the "in-context computation" that ANOVA cannot capture.

**Current state:**
- `fig_natural_r2_major_vs_all.py` exists and produces `natural_r2_major_vs_all_v6.png` and `natural_r2_major_vs_all_v8.png` — these show the comparison **for E3 (latent/Markov) only**.
- No equivalent exists for **E1 (coin)** or **E2 (linear)**.

**Action needed:**
- [ ] Run `fig_natural_r2_major_vs_all.py` for E1 (coin) and E2 (linear) with k=10 and produce similar major-vs-minor comparison plots.
- [ ] Create a summary table of R^2 at the last position for each (task type, layer) for all three experiments, split by major/minor. This would be a powerful table for the paper or appendix.
- [ ] Decide where this goes: main paper (new figure in Sec 5 or 6) or appendix.

#### ~~Priority 2: Natural-sequence R^2 for E1 (coin)~~ — NOT NEEDED

For E1 (coin), tokens are i.i.d. given z — no Markov dependency between consecutive tokens. Intervening on s_t does not break any chain consistency (unlike E3 where the intervention disrupts the bigram structure). The interventional R^2 and natural R^2 are essentially the same for E1. The existing interventional results (`task_vector_r2_coin_k10_minor.png`) already serve this purpose.

#### Priority 3: No task vectors for minor tasks — already indirectly supported

The claim "no task vectors for novel tasks" is already indirectly supported by existing evidence:

- **KL transition plots** (`kl_transition_combined_logx.png`, Fig 3): At high task diversity, the model favors M2 (extrapolative task learning), which by definition does not use stored task representations. The phase transition itself shows the model transitions away from retrieval-based inference.
- **OOD R^2 drops with diversity** (Fig 4): OOD hidden states project poorly onto the task-vector subspace, meaning they do not align with any task vector.
- **OOD trajectory plots** (`latent_traj_aligned.png`, Fig 5): OOD prompts drift outside the task-vector simplex.

No additional experiments are strictly necessary. The theoretical argument in Sec 4.2 is well-supported by these existing results. A minor-task simplex trajectory would strengthen the story but is optional.

#### ~~Priority 4: In-context statistics are encoded in hidden states~~ — ALREADY DONE

This is comprehensively covered in Appendix D.3 (`ortho_intervene.tex`, subsection "Decomposition of Orthogonal Directions", label `app:orth_decomp`). The existing results show:

- **E1 (Dice):** V_opt^T h encodes empirical unigram CLR with R^2 up to 0.88 in later layers. Current token dominates early layers, unigram CLR takes over in deeper layers. Combined R^2 = 0.83-0.94 across all layers.
- **E3 (Markov):** V_opt^T h encodes empirical bigram CLR with R^2 up to 0.75 at layer 5. A developmental transition: layers 0-2 encode something not linearly accessible from token statistics (R^2 ~ 0.01-0.04), then layers 3-5 develop a readable bigram predictor (R^2 ~ 0.61-0.85).
- **E2 (Linear):** V_opt^T h encodes the prediction score x_t · beta_hat with R^2 up to 0.80 in later layers.

Furthermore, **causal filtered interventions** confirm that the linearly accessible content accounts for essentially all of V_opt's causal power (filtered bars match V_opt bars in Figs in appendix).

The analysis uses OOD/minor-task sequences. The V_opt directions are optimized to disrupt minor-task prediction (not major). This directly validates the claim that the orthogonal subspace encodes running statistics for unmemorized tasks.

**No additional experiments needed.** The main text in `ortho_intervention_v2.tex` already references these results ("V_opt^T h linearly encodes prediction-relevant context statistics with high held-out R^2 across layers"). The connection to Sec 4.2's framework could be made more explicit in a text edit.

---

## Section 6 — Coexistence and Geometry of Two Modes

### What exists

- Phase transition plots (KL comparison, Fig 3)
- OOD R^2 drops with diversity (Fig 4)
- Causal interventions (Table 2)
- Trajectory plots (Fig 5)

### What may need updating

1. **Connect to Sec 4.2 more explicitly.** The current Sec 6 text introduces the "near-orthogonal representation hypothesis" somewhat independently. With the revised Sec 4.2 predicting representational decoupling, the Sec 6 results should be framed as validating this prediction. *(Text edit, no new experiments.)*

2. **Minor-task R^2 in the orthogonality context.** Currently, Fig 4 shows OOD R^2 dropping with diversity. Minor tasks are the concrete instantiation of the "generalization mode" within the training distribution. The existing causal intervention results (Table 2) already show that suppressing the orthogonal subspace disrupts minor-task performance while leaving major tasks intact — this is functionally equivalent to showing that minor tasks use a different subspace. *(No new experiments needed, but a text connection to Sec 4.2 would help.)*

---

## Section 7 — Boundary Case (Dyck)

### Current state

The Dyck section shows that non-Markovian tasks violate P0-P1. With the revised Sec 4 emphasizing the Markov property as the foundation for the ANOVA framework, the Dyck section becomes an even better illustration of what happens when the structural assumption breaks.

### What may need updating

- [ ] Add a sentence connecting to the revised Sec 4: "The ANOVA framework in Section 4.1 relies on the Markov property. The Dyck language violates this property..." *(Minor text edit.)*

---

## Summary of Action Items

### New experiments needed

| Priority | Experiment | Tasks | Status |
|----------|-----------|-------|--------|
| **P1** | Major-vs-minor ANOVA R^2 for E1, E2 | Adapt `fig_natural_r2_major_vs_all.py` for coin and linear with k=10 | **TODO** |
| **P1** | Summary table of R^2(last position) by major/minor for E1-E3 | Aggregate results from all three experiments | **TODO** |

### Already covered (no new experiments)

| Original priority | What | Why it's covered |
|---|---|---|
| ~~P2~~ | Natural-sequence R^2 for E1 | Coin is i.i.d.; interventional = natural |
| ~~P3~~ | No task vectors for minor tasks | KL transition plots, OOD R^2, trajectory plots already support this |
| ~~P4~~ | Context-stat encoding in orthogonal subspace | Appendix D.3 (`app:orth_decomp`) comprehensively covers this with probes and causal filtered interventions |

### Text edits needed (no experiments)

| Section | Edit | Priority |
|---------|------|----------|
| Sec 5 (`task_vectors_v3.tex`) | Note that Table 1 results are for the retrieval regime (K=3, all memorized) | **P1** |
| Sec 6 (`main.tex` / two modes text) | Frame orthogonality results as validating Sec 4.2's decoupling prediction | **P1** |
| Sec 6 (`ortho_intervention_v2.tex`) | Connect "Mechanism of Orthogonal Subspace" paragraph more explicitly to Sec 4.2's claim about in-context statistics | **P1** |
| Sec 7 (`motivating.tex`) | Connect Dyck to the Markov-property foundation in revised Sec 4 | **P2** |
| Intro (`intro_v3.tex`) | Mention ANOVA framework and "no task vectors for generalization" | **P2** |

### Figures to produce

| Figure | Description | Where it goes |
|--------|-------------|---------------|
| Major-vs-minor R^2 comparison (E1, E2, E3) | 3-panel plot, each showing major R^2 and minor R^2 curves across positions | Main paper or appendix |
| R^2 summary table (last position, all layers, major vs minor) | Compact table analogous to Table 1 but split by major/minor | Appendix |
