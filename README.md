usage:
- uv run src/main.py submit smoke   # submit a one-question smoke-test batch to Anthropic
- uv run src/main.py submit full   # submit the full GPQA dataset to Anthropic's batch API
- uv run src/main.py <batch_id>    # fetch, score, and save results for an existing batch

paper:
- make / make build   # from repo root
- make arxiv          # build + arxiv-latex-cleaner + tarball

198*30*((158*5+707*25)/2E6)