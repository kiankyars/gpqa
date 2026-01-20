usage:
- uv run src/main.py submit smoke   # submit a one-question smoke-test batch to Anthropic
- uv run src/main.py submit full   # submit the full GPQA dataset to Anthropic's batch API
- uv run src/main.py <batch_id>    # fetch, score, and save results for an existing batch

paper:
- make / make build   # from repo root
- make arxiv          # build + arxiv-latex-cleaner + tarball

# batch API cost (50% off → /2e6): 198×30×((158·5+707·25)/2e6). 30=6 prompts×5 reps; 5,25=$/M in,out; 158,707=avg in,out tokens/run
198*30*((158*5+707*25)/2E6)