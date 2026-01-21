---
configs:
- config_name: default
  data_files:
  - split: train
    path: "gpqa_with_subdomain.jsonl"
---

# GPQA Results: Formatting Constraints on Reasoning

Evaluation results for **Claude Opus 4.5** on **GPQA Diamond** (198 questions) under six formatting constraints. From the study *Formatting Tax: How Constraints Affect Reasoning*.

## Dataset Description

- **Curated by:** Kian Kyars
- **Language:** English
- **License:** MIT
- **Source:** [idavidrein/gpqa](https://huggingface.co/datasets/idavidrein/gpqa) (`gpqa_diamond` split)

## Dataset Structure

5,940 rows: 198 questions × 6 prompts × 5 repetitions. Each row is one model response.

| Column | Type | Description |
|--------|------|-------------|
| `question_id` | int | Index into GPQA Diamond (0–197) |
| `prompt` | int | 0=Baseline, 1=Strict JSON, 2=Structural Rigidity, 3=Python, 4=Oulipo, 5=Banned Words |
| `repetition` | int | Run index (0–4) |
| `extracted_letter` | str | Parsed answer: A, B, C, or D |
| `score` | float | 1 if correct, 0 otherwise |
| `correct_answer` | int | Index of correct choice (0–3) in the shuffled order used for that row |
| `input_tokens` | int | Prompt token count |
| `output_tokens` | int | Completion token count |
| `response_text` | str | Full model output |
| `subdomain` | str | e.g. Physics, Organic Chemistry, Molecular Biology |

## Direct Use

- Replicate or extend accuracy/token analyses.
- Study impact of output-format constraints on GPQA.
- Inspect `response_text` for constraint adherence (see associated code).

## Creation

Model: Claude Opus 4.5 (`claude-opus-4-5-20251101`) with thinking enabled and high effort. Prompts add rules on *reasoning* format only; the answer rule `solution:X` with X∈{A,B,C,D} is shared. Code and prompt definitions: [github.com/kiankyars/gpqa](https://github.com/kiankyars/gpqa).

## Limitations

Single model and benchmark; may not generalize. `correct_answer` is in the shuffled choice order for that row, not the canonical GPQA order.

## Citation

**BibTeX:**

```bibtex
@misc{kyars2025formatting,
  author = {Kian Kyars},
  title = {Formatting Tax: How Constraints Affect Reasoning},
  year = {2025},
  url = {https://huggingface.co/datasets/kyars/gpqa-results}
}
```

**GPQA benchmark:**

```bibtex
@article{rein2023gpqa,
  title={GPQA: A Graduate-Level Google-Proof Q\&A Benchmark},
  author={Rein, David and others},
  journal={arXiv:2311.12022},
  year={2023}
}
```
