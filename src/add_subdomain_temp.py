"""
Temporary script: build question_id -> Subdomain from gpqa_diamond, add subdomain
to each record in data/gpqa.jsonl, write data/gpqa_with_subdomain.jsonl.

Run: uv run python -m src.add_subdomain_temp

gpqa_diamond row order matches question_id (0..197) as in main.create_batch_requests.
"""

import json
from pathlib import Path

from datasets import load_dataset


def main():
    data_dir = Path("data")
    dataset = load_dataset("idavidrein/gpqa", "gpqa_diamond", cache_dir=str(data_dir))["train"]

    if "Subdomain" not in dataset.column_names:
        raise SystemExit(f"gpqa_diamond has no 'Subdomain'. Columns: {dataset.column_names}")

    qid_to_subdomain = {i: row["Subdomain"] for i, row in enumerate(dataset)}
    if len(qid_to_subdomain) != 198:
        raise SystemExit(f"Expected 198 questions, got {len(qid_to_subdomain)}")

    inp = data_dir / "gpqa.jsonl"
    out = data_dir / "gpqa_with_subdomain.jsonl"
    if not inp.exists():
        raise SystemExit(f"Missing {inp}")

    count = 0
    with open(inp) as f, open(out, "w") as o:
        for line in f:
            rec = json.loads(line)
            qid = rec.get("question_id")
            rec["subdomain"] = qid_to_subdomain.get(qid)
            json.dump(rec, o)
            o.write("\n")
            count += 1

    print(f"Wrote {out} with subdomain for {count} records ({len(qid_to_subdomain)} questions).")


if __name__ == "__main__":
    main()
