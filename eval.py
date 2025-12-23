from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator

class AccuracyEvaluator(Evaluator):
    """Custom evaluator for accuracy calculation."""
    def evaluate(self, ctx):
        # Comparison logic for pydantic-evals
        return 1.0 if ctx.output == ctx.expected_output else 0.0

def create_pydantic_evals_dataset(processed_results, examples):
    """Create Pydantic Evals dataset from processed results."""
    cases = []
    aggregated_stats = {}
    
    results_by_case = {}
    for result in processed_results:
        key = (result['question_id'], result['prompt'])
        if key not in results_by_case:
            results_by_case[key] = []
        results_by_case[key].append(result)
    
    for (question_id, prompt), results_list in results_by_case.items():
        example = examples[question_id]
        expected_output = example.correct_index
        
        correct_count = sum(1 for r in results_list if r['score'] == 1.0)
        total_count = len(results_list)
        
        cases.append(Case(
            name=f"q{question_id}_p{prompt}",
            inputs={'question': example.question, 'prompt': prompt},
            expected_output=str(expected_output),
            metadata={'question_id': question_id, 'prompt': prompt},
        ))
        
        if prompt not in aggregated_stats:
            aggregated_stats[prompt] = {'correct': 0, 'total': 0}
        aggregated_stats[prompt]['correct'] += correct_count
        aggregated_stats[prompt]['total'] += total_count
    
    dataset = Dataset(cases=cases)
    dataset.add_evaluator(AccuracyEvaluator())
    return dataset, aggregated_stats

dataset, aggregated_stats = create_pydantic_evals_dataset(processed_results, examples)
for level in sorted(aggregated_stats.keys()):
    stats = aggregated_stats[level]
    accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

import logfire
from pydantic_evals import Dataset, Case
from typing import Any

# Configure Logfire to send data
logfire.configure(send_to_logfire='if-token-present')

# 1. Convert your loaded GPQA examples into Eval Cases
# We store the "question_id" in metadata to link back to the batch result
cases = [
    Case(
        input=ex.question,
        expected_output=ex.correct_index, # Store index (0-3) or letter
        metadata={"id": i, "choices": [ex.choice1, ex.choice2, ex.choice3, ex.choice4]}
    )
    for i, ex in enumerate(examples)
]

dataset = Dataset(cases=cases)

# 2. Create a "Lookup Task"
# This simulates an Agent run but uses your pre-computed batch results
def batch_lookup_task(input_data: Any, case: Case) -> str:
    # Find the result for this question_id and specific prompt/repetition
    # (You might loop this whole block over your different prompt levels)
    q_id = case.metadata["id"]
    
    # Example: Look up result for Prompt 0, Repetition 0
    # In reality, you would run 'dataset.evaluate' multiple times, 
    # once for each prompt/repetition combo you want to visualize.
    result_row = next(
        (r for r in processed_results 
         if r['question_id'] == q_id and r['prompt'] == 0 and r['repetition'] == 0),
        None
    )
    
    if not result_row:
        return "MISSING"
    
    # Return the raw text or extracted letter for the evaluator to check
    return result_row['extracted_letter'] # or result_row['response_text']

# 3. Define a Scorer
def exact_match_scorer(case: Case, output: str) -> bool:
    # Map index to letter
    correct_letter = "ABCD"[case.expected_output]
    return output == correct_letter

# 4. Run the Eval (Instant & Free)
# This sends the data to Logfire's "Evals" tab
logfire.info("Uploading batch results to Logfire Evals UI...")
dataset.evaluate_sync(
    batch_lookup_task, 
    scorers=[exact_match_scorer],
    experiment_name="gpqa-batch-prompt-0"
)