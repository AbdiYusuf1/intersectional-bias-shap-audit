# Explaining Intersectional Bias in AI Recruitment Using SHAP

A controlled name injection audit of automated CV screening models.

**Abdihafid Yusuf** — MSc Data Science and Artificial Intelligence, Sheffield Hallam University

---

## Research question

> How successfully can SHAP explain the internal logic of intersectional racial and
> gender bias in AI-based CV screening systems, and can these results distinguish
> genuine fairness from superficial neutrality?

## Overview

Most published fairness audits are **observational**: they compare outcomes across
demographic groups that differ in many other respects, so any disparity is confounded
with genuine differences between candidates.

This project uses an **experimental** design drawn from the correspondence audit
tradition (Bertrand & Mullainathan, 2004). Each CV is duplicated across four
race gender conditions, byte identical apart from the injected name. Comparisons are
made *within* matched sets, so every candidate characteristic is held constant by
construction and any difference in model output is attributable to the name alone.

## Findings

1. **Bag of words screening models are structurally incapable of direct name
   discrimination.** Only 1 of 34 injected name tokens entered the vocabulary of a
   model trained on nameless CVs.

2. **Naive auditing produces false positives.** 391 of 400 scores shifted when a name
   was added — caused by document-length normalisation, not the name itself.

3. **No reproducible name based bias** was found, across two model families and four
   independent methods. Minimum disparate impact ratio 0.968 (threshold 0.80).

4. **Statistical significance within a run is not reproducibility across runs.** A
   single run produced a large, significant and theoretically coherent racial
   disparity (*d* = -0.927, *p* < 0.0001) that reversed sign under a different random
   seed. Seed-stability testing identified it as noise.

## Repository contents

| File | Description |
|---|---|
| `shap_intersectional_audit.ipynb` | Main study: Designs A and B, paired analysis, fairness metrics, SHAP, validation layer, seed stability |
| `rf_cross_model_audit.ipynb` | Cross model replication on Random Forest |
| `results/` | Exported result tables (CSV) |
| `figures/` | Design diagrams and result figures |

## Dataset

[`cnamuangtoun/resume-job-description-fit`](https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit)
(Hugging Face) — 8,000 resume job description pairs labelled *Good Fit*,
*Potential Fit*, or *No Fit*.

**Licence:** none declared. The dataset has no dataset card and no licence tag. It is
published openly for public download and is used here solely for non-commercial
academic research. The dataset is **not redistributed in this repository** — download
it from the link above.

Resume text consists of real CV examples de-identified at source: candidate names
removed and email addresses replaced with placeholders.

## Reproducing the analysis

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy shap
```

1. Download `train.csv` and `test.csv` from the dataset link above
2. Place them in the same directory as the notebooks
3. Run `shap_intersectional_audit.ipynb` first, then `rf_cross_model_audit.ipynb`

All random seeds are fixed (`RANDOM_STATE = 42`). Section 8.2 and 10.1 of the main
notebook repeat the analysis across multiple seeds; reduce `N_PAIRED_SEEDS` and
`N_SEEDS` if runtime is a constraint.

## Methodological safeguards

Each safeguard exists because a specific failure mode was identified during
development:

| Safeguard | Failure mode it prevents |
|---|---|
| Label learnability check | An unlearnable target produces a guaranteed null that mimics fairness |
| Name collision screening | 12% of CVs already contain candidate name tokens, contaminating the manipulation |
| Byte identity assertion | Silent contamination would invalidate the causal claim |
| Exposure matched placebo | Unequal name exposure inflates apparent importance (1.18×) |
| Effect sizes with Holm correction | Significance alone does not indicate magnitude |
| Seed stability testing | Single-run effects can reverse sign across seeds |

## Limitations

- Dataset label provenance is undocumented; labels appear algorithmically derived
  rather than confirmed human hiring decisions
- Resume text is drawn from public CV templates, not live applications
- Names are US validated (Bertrand & Mullainathan, 2004; 1974–79 records) applied in a
  UK and EU regulatory framing
- Bag of words representations cannot capture contextual name effects that
  transformer-based screening models might exhibit

## References

Bertrand, M., & Mullainathan, S. (2004). Are Emily and Greg more employable than
Lakisha and Jamal? A field experiment on labor market discrimination.
*American Economic Review, 94*(4), 991–1013.

Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model
predictions. *Advances in Neural Information Processing Systems, 30*.

Slack, D., Hilgard, S., Jia, E., Singh, S., & Lakkaraju, H. (2020). Fooling LIME and
SHAP: Adversarial attacks on post hoc explanation methods. *Proceedings of the
AAAI/ACM Conference on AI, Ethics, and Society*, 180–186.

Wood, M., Hales, J., Purdon, S., Sejersen, T., & Hayllar, O. (2009). *A test for racial
discrimination in recruitment practice in British cities* (Research Report 607).
Department for Work and Pensions.

---
