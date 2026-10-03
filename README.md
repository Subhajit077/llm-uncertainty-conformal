# Uncertainty and conformal prediction for LLMs under distribution shift

Two questions, studied on Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct-AWQ:
1. Which signals best detect wrong answers in closed-book QA (TriviaQA, n=5000)?
2. Does split-conformal prediction on MMLU (n=14,042, 57 subjects) keep its coverage under subject shift, and how many target labels repair it?

## Findings

**Error detection (TriviaQA, AUROC; 95% bootstrap CIs are in the notebooks).**

| Method | 3B | 7B |
|---|---|---|
| Token probability | 0.835 | 0.849 |
| Self-consistency (10 samples) | 0.826 | 0.840 |
| Semantic entropy (10 samples + NLI) | 0.827 | 0.828 |
| Logistic combination (5-fold CV) | 0.846 | 0.850 |

- Greedy accuracy on TriviaQA: 45.0% (3B), 52.0% (7B).
- Free token probability is the strongest single signal for both models. Paired bootstrap: semantic entropy is slightly worse than token probability (3B diff CI [-0.016, -0.002]; 7B [-0.028, -0.014]).
- Combining signals helps at 3B (paired CI [+0.006, +0.015]) but not at 7B ([-0.002, +0.006]).

**Conformal prediction (LAC score, target coverage 90%, MMLU).**

| Setting | 3B | 7B |
|---|---|---|
| MMLU accuracy | 62.7% | 69.9% |
| iid split: coverage / set size | 0.899 / 2.24 | 0.900 / 1.82 |
| Calibrate on easy subjects, test on hard | 0.848 | 0.753 |
| + recalibrate on k=50 target labels | 0.904 (sd 0.042) | 0.899 (sd 0.039) |
| + recalibrate on k=200 | 0.902 (sd 0.022) | 0.900 (sd 0.020) |

- Coverage holds on average under random splits but fails under a harsh subject shift, by more for 7B. This contrast is suggestive only: "easy" and "hard" are defined by each model's own accuracy, so the splits differ.
- A small labeled target sample restores average coverage, at the cost of larger prediction sets (about 2.3 to 2.8 of 4 options). Single recalibrations with small k still vary by several points.
- Per-subject thresholds raise worst-subject coverage modestly but are limited by small group sizes.

## Limitations
One dataset per task, one model family, short answers only. Correctness is judged by alias matching. About 2% of 7B MMLU rows lack an option letter in the top-20 logprobs. "Worst subject" statistics are biased downward by taking a minimum over noisy estimates. Not peer reviewed.

## Reproduce
Notebooks: `01_uncertainty_conformal_qwen7b.ipynb` (full pipeline, vLLM on Kaggle T4 x2) and `01_uncertainty_qwen3b.ipynb`. Data: `triviaqa_*_scored.json`, `mmlu_*.npz`. Figures to be added.
