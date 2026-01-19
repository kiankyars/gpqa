"""Fix answer extraction for existing JSONL files that have JSON format responses."""
import json
import re
from pathlib import Path


def extract_answer_from_response(response_text, prompt):
    """Extract answer from response, handling both JSON and plain text formats."""
    extracted_letter = None
    
    # For JSON format (prompt 1 - Strict JSON)
    if prompt == 1:
        try:
            # Try to find JSON block (may be wrapped in ```json or just raw JSON)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly (may span multiple lines)
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*"solution"[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # Try to parse entire response as JSON
                    json_str = response_text.strip()
            
            parsed_json = json.loads(json_str)
            if 'solution' in parsed_json:
                extracted_letter = str(parsed_json['solution']).strip().upper()
        except (json.JSONDecodeError, AttributeError, KeyError):
            pass
    
    # Fallback to plain text format (for other prompts or if JSON parsing failed)
    if extracted_letter is None:
        match_ans = re.search(r'solution:\s*([A-D])', response_text, re.IGNORECASE)
        if match_ans:
            extracted_letter = match_ans.group(1).upper()
    
    return extracted_letter


def fix_jsonl_file(filepath):
    """Fix answer extraction in a single JSONL file."""
    print(f"Processing {filepath}...")
    
    fixed_records = []
    fixed_count = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            original_extracted = record.get('extracted_letter')
            
            # Re-extract answer
            new_extracted = extract_answer_from_response(record['response_text'], record['prompt'])
            
            if new_extracted != original_extracted:
                fixed_count += 1
                record['extracted_letter'] = new_extracted
                
                # Recalculate score
                correct_letter = 'ABCD'[record['correct_answer']]
                record['score'] = 1.0 if new_extracted == correct_letter else 0.0
            
            fixed_records.append(record)
    
    # Write fixed records back
    backup_path = filepath.with_suffix('.jsonl.backup')
    print(f"  Creating backup: {backup_path}")
    Path(filepath).rename(backup_path)
    
    with open(filepath, 'w') as f:
        for record in fixed_records:
            f.write(json.dumps(record) + '\n')
    
    print(f"  Fixed {fixed_count} records")
    return fixed_count


def main():
    """Fix all JSONL files in data directory."""
    data_dir = Path("data")
    jsonl_files = list(data_dir.glob("results_*.jsonl"))
    
    if not jsonl_files:
        print("No JSONL files found in data/ directory")
        return
    
    print(f"Found {len(jsonl_files)} JSONL file(s)")
    total_fixed = 0
    
    for filepath in jsonl_files:
        fixed = fix_jsonl_file(filepath)
        total_fixed += fixed
    
    print(f"\nTotal records fixed: {total_fixed}")
    print("Backups created with .backup extension")


if __name__ == "__main__":
    main()
