usage:
- uv run src/main.py submit smoke   # submit a one-question smoke-test batch to Anthropic
- uv run src/main.py submit full   # submit the full GPQA dataset to Anthropic's batch API
- uv run src/main.py <batch_id>    # fetch, score, and save results for an existing batch

paper:
- make / make build   # from repo root
- make arxiv          # build + arxiv-latex-cleaner + tarball

### Batch API cost (50% off → /2e6)

```
198 * 30 * ((344*5 + 11529*25) / 2e6)
```

| Symbol | Value | Meaning |
|--------|-------|---------|
| 198 | questions | GPQA Diamond |
| 30 | 6×5 | prompts × reps |
| 344 | tokens/run | mean input (from data) |
| 11529 | tokens/run | mean output (from data) |
| 5, 25 | $/M | Opus 4.5 input, output |
| 2e6 | | per‑million × 2 (batch discount) |