"""Extract answer from model response; handles JSON (prompt 1) and plain 'solution: X'."""
import json
import re


def extract_answer_from_response(response_text: str, prompt: int) -> str | None:
    """Extract A-D from response. For prompt 1 tries JSON first; else uses solution: X."""
    out = None
    if prompt == 1:
        try:
            m = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            s = m.group(1) if m else None
            if s is None:
                m2 = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"solution"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                s = m2.group(0) if m2 else response_text.strip()
            obj = json.loads(s)
            if 'solution' in obj:
                out = str(obj['solution']).strip().upper()
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
            pass
    if out is None:
        m = re.search(r'solution:\s*([A-D])', response_text, re.IGNORECASE)
        out = m.group(1).upper() if m else None
    return out
