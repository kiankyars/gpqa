"""Lean manual checks that each prompt's constraint is respected in results JSONL.

Uncertainties:
- Oulipo (p4): we exclude the final "solution: X" line when checking for 'e', since
  "solution" contains e. The prompt says "reasoning chain"; if the solution line
  is meant to be included, the constraint would be impossible.
- Structural (p2): we only check ≥3 bullets, not "because"/"therefore".
- Strict JSON (p1): we only check that some JSON parses, not the 5 required keys.
"""
import json
import re
import sys
from pathlib import Path

BANNED = "accordingly, code, ocode, The, settings, Moreover, description, Let's, This, core, utilizes, revolves, Here's, possibly, logic, thereby".split(", ")

# Short names for the paper's table
PROMPT_NAMES = {
    0: "Baseline",
    1: "Strict JSON",
    2: "Structural Rigidity",
    3: "Python Code",
    4: "Oulipo",
    5: "Banned Words",
}


def check_p0(text):
    return "solution:" in text


def check_p1_strict_json(text):
    # any parseable JSON (lean: has { } and json.loads works on first candidate)
    m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if not m:
        return False
    try:
        json.loads(m.group(0))
        return True
    except json.JSONDecodeError:
        return False


def check_p2_structural(text):
    # at least 3 bullet lines (•, -, * at line start)
    bullet = re.compile(r'^[\s]*([\-\*•])\s', re.M)
    return len(bullet.findall(text)) >= 3


def check_p3_python(text):
    return "```python" in text


def check_p4_oulipo(text):
    # no 'e' or 'E' in reasoning; exclude final "solution: X" line
    before = re.split(r'\nsolution:\s*[A-Da-d]\s*$', text, flags=re.I)[0]
    return 'e' not in before and 'E' not in before


def check_p5_banned(text):
    for w in BANNED:
        if re.search(r'\b' + re.escape(w) + r'\b', text):
            return False
    return True


CHECKS = {
    0: ("Baseline (has solution:)", check_p0),
    1: ("Strict JSON", check_p1_strict_json),
    2: ("Structural (≥3 bullets)", check_p2_structural),
    3: ("Python (```python)", check_p3_python),
    4: ("Oulipo (no e/E before solution:)", check_p4_oulipo),
    5: ("Banned words (none of 16)", check_p5_banned),
}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/gpqa.jsonl"
    rows = [json.loads(ln) for ln in open(path)]
    by_p = {}
    for r in rows:
        p = int(r["prompt"])
        by_p.setdefault(p, []).append(r)

    # JSON for legacy / programmatic use
    out = {}
    for p in sorted(CHECKS):
        _, fn = CHECKS[p]
        lst = by_p.get(p, [])
        ok = sum(1 for r in lst if fn((r.get("response_text") or "")))
        out[str(p)] = [ok, len(lst)]

    json_path = Path("data/test.json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(out, f, separators=(',', ':'))

    # LaTeX table to figures/
    repo_root = Path(__file__).resolve().parent.parent
    tex_path = repo_root / "figures" / "compliance_table.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\caption{Format compliance by prompt (\% of responses that followed the constraint)}",
        r"\label{tab:compliance}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Prompt & Condition & \% Followed & N \\",
        r"\midrule",
    ]
    for p in sorted(CHECKS):
        lst = by_p.get(p, [])
        ok = out[str(p)][0]
        n = len(lst)
        pct = (100.0 * ok / n) if n else 0.0
        name = PROMPT_NAMES.get(p, CHECKS[p][0])
        # Escape _ in condition names for LaTeX
        name = name.replace("_", r"\_")
        lines.append(f"{p} & {name} & {pct:.1f} & {n} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
