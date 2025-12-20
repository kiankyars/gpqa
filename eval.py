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