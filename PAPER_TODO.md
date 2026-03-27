# Paper TODO: Representation Geometry of Task Inference in Transformers

This document audits every `.tex` file in `representation_geometry_two_modes/` and flags
outstanding gaps — missing content, unresolved author comments, broken references, and
open design decisions — with recommendations on how to approach each.

---

## 1. Sections With No Content (Critical)

### 1.1 Related Work (`related.tex`)
**Status: shell only — no body text.**

The file has a section header and three empty paragraph headings:
```
\paragraph{Task vectors in language models.} \YZ{briefly mention the common practice...}
\paragraph{Inference modes and generalization behaviors.}
\paragraph{Near-orthogonal representation geometry.}
```
None of the three paragraphs has any prose.

**How to approach:**
Each paragraph maps naturally to a cluster of papers you are already citing in the main
text. Draft each one in sequence:

- **Task vectors** — cite Hendel et al. 2023, Yang et al. 2025, Mikolov embeddings,
  Park et al. linear representation hypothesis. Two to three sentences describing the
  common practice of extracting middle-layer task vectors and how they steer behavior.
- **Inference modes** — cite the pan2023context, park2024competition you already
  reference in `intro.tex`; mention Xie et al. Bayesian ICL, Garg et al., Olsson/Edelman
  induction-head work. Note what is still unexplained (geometry, training data effects).
- **Near-orthogonal geometry** — cite Elhage superposition (2022), any follow-up work on
  orthogonal task encoding; tie to why this matters for coexistence of modes.

Target: ~4–6 sentences per paragraph, 0.5–0.75 pages total.

---

### 1.2 Limitations and Future Work (`main.tex`, §Limitations)
**Status: section header exists, zero content.**

The section appears immediately before `\bibliographystyle` with nothing between them.

**How to approach:**
Write 3–4 concise bullet points or a short paragraph on each of:
1. **Scope of synthetic settings** — small transformers trained from scratch; unclear
   how well geometry transfers to pre-trained LLMs at scale.
2. **Markov assumption** — task-vector theory rests on the Markov property (Section 2
   Remark); E4 (Dyck) shows this is not universal.
3. **OOD task sampling** — OOD tasks are drawn from the same prior as minor tasks;
   genuinely novel task families (different function class) are not tested.
4. **Future directions** — theoretical characterization of the phase-transition boundary;
   probing pre-trained LLMs; causal story for why diversity induces orthogonalization.

This is the easiest gap to fill; write freely and trim later.

---

## 2. Unresolved Author Comments (`\YZ{...}` / `\HY{...}`)

### 2.1 `intro.tex` — Figure 1 caption
```latex
\YZ{please help with the figure.}
```
The figure file `main-Figs/main-drawio.png` **already exists**. The caption only needs
a prose description.

**Fix:** Write 2–3 sentences explaining what the figure shows: a schematic of the rolling
biased dice experiment in which a sequence with latent z is fed to a transformer and the
middle layer produces a task vector. Remove the `\YZ` marker.

---

### 2.2 `task_vectors_v2.tex` — Extraction sentence and footnote
```latex
\YZ{Briefly mention how $\vh_t$ is extracted from the transformer.}
```
This is a request for one bridging sentence right before Definition 1. The extraction
procedure is **fully described** in `Appendix/task_vec_extract.tex` (§sec:extracting-hiddens).

**Fix:** Add one sentence such as:
> "We extract $\vh_t^{(\ell)}$ from the residual stream after the attention sub-block of
> layer $\ell$ (see Appendix X for details)."

Then replace `Appendix~??` in the footnote with the correct label `\ref{app:task-vec-extract}`.

Also remove the comment `\YZ{Haolin: could you help me with this part?}` once the sentence
is written.

---

### 2.3 `task_vectors_v2.tex` — Duplicate ANOVA sentence
Lines 99 and 103 contain the **identical sentence** about the ANOVA decomposition:
```
We evaluate the approximation using analysis of variance (ANOVA), where...
```
One of the two copies should be deleted (the commented-out block between them suggests the
first copy was left in by accident when the paragraph was rewritten).

---

### 2.4 `two_ modes.tex` — Missing citations (`\YZ{cite}`)
Two citation placeholders remain:
- Line 11: prior work on task diversity driving inference-mode transitions.
- Line 20: same cluster of papers.

**Fix:** Fill with the relevant citations already in `refs.bib`. Based on context (task
diversity → mode shift) the likely targets are Raventós et al. 2023 (`raventos2023pretraining`),
Garg et al. 2022, and/or Pan et al. 2023. Check which ones you intended and add the keys.

---

### 2.5 `two_ modes.tex` — Unresolved prior choice
```latex
\YZ{Do we use uniform prior or majority-minority prior?}
\HY{This is still majority-minority prior}
```
The answer is known (majority-minority), but both comments remain in the source.

**Fix:** Remove both comment macros. If the answer affects the figure caption (it does —
line 43 says "uniform prior" in the caption text), update the caption to say
"majority-minority prior" consistently.

---

### 2.6 `ortho_intervention.tex` — Design decision on minority results
```latex
\YZ{Do we remove minority intervention from the main text?
    Putting it in the appendix and adding explanations there?}
```
The table currently shows three columns (Maj, OOD, Min) for both interventions. The minority
column has interesting small-negative values for E3 and E2 that are explained in the blue
text. The blue commentary itself (`{\color{blue}...}`) also flags it as tentative.

**Decision required (authors):** Keep in main text with a sentence of explanation, or move
to appendix with a pointer. Whichever is chosen, the blue `\color{blue}` text must be
converted to normal text or cut.

---

### 2.7 `ortho_intervention.tex` — Rank of orthogonal subspace
```latex
\YZ{What is the rank of the orthogonal subspace. Say that the orthogonal subspace is
    discovered via XXX optimization...}
```
The orthogonal intervention appendix (`ortho_intervene.tex`) gives full algorithmic details
including that `s = 1` direction is optimised. The rank of `U_perp` is `D − p` where
`p` depends on task and token protection components.

**Fix:** Report `s = 1` (one optimised direction) and state that the direction is found via
the Adam optimisation described in Appendix \ref{app:orth_ablation}. Then remove the
`\YZ` marker.

---

### 2.8 `synthetic_setup.tex` — Hyperparameter note
```latex
%We use autoregressive training in all experiments. We choose hyperparameters \YZ{to complete.}
```
This is a commented-out line — the paragraph directly below it correctly references
`Appendix~\ref{app:hyperparams}` and says "A complete list is provided there." The comment
is therefore obsolete.

**Fix:** Delete the commented-out line.

---

## 3. Factual / Technical Inconsistency

### 3.1 Layer count for E3 vs. E2 (`synthetic_setup.tex` vs. `experiment_hyperparams.tex`)

`synthetic_setup.tex` says:
> "for \texttt{E3}, we follow \citet{akyurek2022learning} by increasing the number of
> layers to $16$"

`Appendix/experiment_hyperparams.tex` (Table 1) shows:
| Task | Layers |
|---|---|
| Biased dice (E1) | 6 |
| Markov chains (E3) | **6** |
| Planted Dyck (E4) | 6 |
| Linear regression (E2) | **16** |

**The text says E3 has 16 layers; the table says E2 (linear regression) has 16 layers.**
This is a contradiction. Akyürek et al. 2022 is about in-context linear regression, so
the 16-layer design likely belongs to E2.

**Fix:** Change the sentence in `synthetic_setup.tex` to read "for \texttt{E2}" (linear
regression), not "\texttt{E3}" (Markov chains). Verify this against actual training runs.

---

## 4. Stale TODO Comments (Minor / Cosmetic)

The following `% TODO` comments were left in the source after the figures were already
generated and placed in `Figs/`. They are **not blocking** but should be cleaned up before
submission.

| File | Lines | Comment |
|---|---|---|
| `task_vectors_v2.tex` | 138, 144, 150 | `% TODO: replace with X averaging R^2 figure` |
| `task_vectors_v2.tex` | 170, 176, 182 | `% TODO: replace with X trajectory figure` |
| `id_ood_loss.tex` | 16, 22, 28 | `% TODO: replace with ID/OOD loss plot for Ex` |

All referenced figures (`coin_averaging_r2.png`, `latent_beta_alpha_traj.png`,
`coin_id_ood_loss.png`, etc.) **exist** in `representation_geometry_two_modes/Figs/`.

**Fix:** Delete the `% TODO` comment lines. No other change needed.

---

## 5. Missing Citations in `Dyck.tex`

```latex
A commonly held belief is that transformers can often compress contexts into hidden
states \citep{}, which effectively summarize the preceding context... \citep{}
```
Two `\citep{}` calls with no citation keys on line 6.

**Fix:** Fill in appropriate references. Candidates already in `refs.bib`:
`hendel2023context`, `yang2025task`, or any probing / representation-compression paper
you intend to cite.

---

## 6. Open Design Questions (Require Author Decision)

These are not writing gaps but unresolved scientific or presentation choices flagged in
comments:

| Location | Question |
|---|---|
| `ortho_intervention.tex` §conclusions | `\YZ{Try to save space...}` — the conclusions paragraph may be too long; decide what to cut. |
| `task_vectors_v2.tex` (commented block) | `\YZ{Do we need to state this in the main text, or in the appendix??}` — a Hao conceptual formulation of $\vh_t$ as a concatenation of basis vectors; decide whether to include or drop. |
| `task_vectors_v2.tex` (commented block) | `\YZ{We can state a toyish theorem here...}` — possible mini-theorem connecting P2+P3 to the Bayesian formula; decide if in-scope. |

---

## 7. Figures: Orphaned / Legacy Files in `Figs/`

The `Figs/` directory contains ~150 files; only ~30–40 are actually referenced by `.tex`
files currently included in `main.tex`. There are many exploratory/legacy figures
(e.g., `coins_0_traj_mean.png`, `coin_alpha_inject_0.png`,
`historical_latent_0.png`, `reverse_probe_coin_k-1.png`, etc.) that are not referenced
anywhere in the current paper.

**These are not bugs**, but before final submission it is worth doing a pass to:
1. Confirm every figure included in a `\includegraphics` is in `Figs/` — **all currently
   referenced figures are present**.
2. Optionally delete/archive unreferenced legacy figures.

---

## 8. Summary Priority Table

| Priority | Item | Effort | File |
|---|---|---|---|
| 🔴 High | Write related work body | ~2–3 hours | `related.tex` |
| 🔴 High | Write limitations section | ~30 min | `main.tex` |
| 🔴 High | Fix E2/E3 layer count inconsistency | 5 min | `synthetic_setup.tex` |
| 🟠 Medium | Fill two `\YZ{cite}` placeholders | 10 min | `two_ modes.tex` |
| 🟠 Medium | Write figure 1 caption | 15 min | `intro.tex` |
| 🟠 Medium | Add extraction sentence + fix `Appendix~??` | 10 min | `task_vectors_v2.tex` |
| 🟠 Medium | Decide: minority results in main or appendix | Author decision | `ortho_intervention.tex` |
| 🟠 Medium | Resolve uniform vs. majority-minority prior note | 5 min | `two_ modes.tex` |
| 🟡 Low | Add 2 missing `\citep{}` keys | 5 min | `Dyck.tex` |
| 🟡 Low | Remove duplicate ANOVA sentence | 2 min | `task_vectors_v2.tex` |
| 🟡 Low | Remove stale `% TODO` comments | 5 min | `task_vectors_v2.tex`, `id_ood_loss.tex` |
| 🟡 Low | Delete commented-out hyperparameter note | 1 min | `synthetic_setup.tex` |
| 🟡 Low | Add rank/optimization sentence for orth. subspace | 5 min | `ortho_intervention.tex` |
| 🟡 Low | Convert blue tentative text to normal or cut | 10 min | `ortho_intervention.tex` |

---

## 9. Recommended Order of Attack

1. **Fix the layer-count inconsistency first** (§3.1) — it is a factual error and affects
   experimental credibility. Confirm against your actual training code which experiment
   uses 16 layers.

2. **Write Related Work** (§1.1) — this is the only section that is truly blank and will
   block any co-author review or submission. Do a first draft quickly and iterate.

3. **Write Limitations** (§1.2) — short, unconstrained, easy to write. Do it immediately
   after related work while your thinking is warmed up.

4. **Batch-fix all small markers** — do a global search for `\YZ{`, `\HY{`, `% TODO`,
   `\citep{}`, `Appendix~??` and resolve each one in a single editing session. Most
   take under 5 minutes individually.

5. **Make the minority-in-appendix decision** (§2.6) — this requires a co-author
   conversation but is architecturally simple once decided.

6. **Final figure and reference audit** — compile the PDF and check for broken `\ref`,
   missing figures, and double-blind compliance before submission.
