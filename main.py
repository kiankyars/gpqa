"""
Bureaucratic Jam Hypothesis Experiment

Tests whether formatting constraints degrade reasoning capabilities by running
198 GPQA questions through 5 constraint levels with 5 repetitions each.
"""

import json
import os
import random
import re
import csv
from collections import namedtuple
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

import logfire
from anthropic import Anthropic
from datasets import load_dataset
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from tqdm import tqdm

# Initialize Logfire
logfire.configure()

# Initialize Anthropic client
anthropic = Anthropic()

Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

LETTER_TO_INDEX = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
CONSTRAINT_LEVELS = 5
REPETITIONS = 5
NUM_QUESTIONS = 198
MODEL = "claude-opus-4-5-20251101"


def load_gpqa_dataset() -> List[Example]:
    """Load GPQA dataset from Hugging Face and convert to Example format."""
    logfire.info("Loading GPQA dataset from Hugging Face")
    
    # Load dataset - may need to accept terms
    dataset = load_dataset("idavidrein/gpqa", split="test")
    
    # Filter to first 198 questions (Diamond subset)
    questions = dataset.select(range(min(NUM_QUESTIONS, len(dataset))))
    
    examples = []
    random.seed(0)  # Consistent shuffling
    
    for item in questions:
        # Extract choices - handle different possible column names
        # GPQA dataset structure: Question, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3, Correct Answer
        question = item.get('Question', item.get('question', item.get('Question', '')))
        
        # Try different possible column name formats
        incorrect1 = (item.get('Incorrect Answer 1') or 
                     item.get('incorrect_answer_1') or 
                     item.get('Incorrect Answer 1', ''))
        incorrect2 = (item.get('Incorrect Answer 2') or 
                     item.get('incorrect_answer_2') or 
                     item.get('Incorrect Answer 2', ''))
        incorrect3 = (item.get('Incorrect Answer 3') or 
                     item.get('incorrect_answer_3') or 
                     item.get('Incorrect Answer 3', ''))
        correct = (item.get('Correct Answer') or 
                  item.get('correct_answer') or 
                  item.get('Correct Answer', ''))
        
        # Skip if missing required fields
        if not all([question, incorrect1, incorrect2, incorrect3, correct]):
            logfire.warn(f"Skipping item with missing fields: {item}")
            continue
        
        # Shuffle choices
        list_choices = [incorrect1, incorrect2, incorrect3, correct]
        random.shuffle(list_choices)
        correct_index = list_choices.index(correct)
        
        example = Example(
            question=question,
            choice1=list_choices[0],
            choice2=list_choices[1],
            choice3=list_choices[2],
            choice4=list_choices[3],
            correct_index=correct_index
        )
        examples.append(example)
    
    logfire.info(f"Loaded {len(examples)} examples from GPQA dataset")
    return examples


def load_from_json(path: str) -> Dict:
    """Load JSON from a file."""
    with open(path, 'r') as f:
        return json.load(f)


def generate_prompt_from_examples(json_data: Dict, with_explanations: bool = True) -> str:
    """Generate prompt text from example JSON data."""
    output = ""
    for q in json_data["questions"]:
        output += f'Question: {q["question"]}\nChoices:\n'
        for choice, value in q["choices"].items():
            output += f'({choice}) {value}\n'
        
        if with_explanations:
            output += f"Let's think step by step: \n{q['explanation']}\n"
        
        output += f'The correct answer is ({q["correct_answer"]})\n'
    
    return output


def chain_of_thought_prompt(example: Example) -> str:
    """Creates a chain-of-thought prompt given a single example."""
    prompt = "Here are some example questions from experts. An explanation is given before the final answer. Answer the final question yourself, giving your reasoning beforehand.\n"
    
    # Try to load examples JSON if it exists
    examples_path = "prompts/chain_of_thought_examples.json"
    if os.path.exists(examples_path):
        json_data = load_from_json(examples_path)
        prompt += generate_prompt_from_examples(json_data, with_explanations=True)
    
    prompt += f"Question: {example.question}"
    prompt += f"\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}"
    prompt += "\nGive step by step reasoning before you answer, and when you're ready to answer, please use the format \"The correct answer is (insert answer here)\":\n"
    return prompt


def create_prompt_level_1(example: Example) -> str:
    """Level 1 - Baseline (Control): Chain of thought prompt."""
    return chain_of_thought_prompt(example)


def create_prompt_level_2(example: Example) -> str:
    """Level 2 - Agent (Strict JSON)."""
    base = chain_of_thought_prompt(example)
    constraint = "\n\nIMPORTANT: You must output valid JSON only. The JSON must contain exactly these five keys: `initial_intuition`, `step_by_step_logic`, `potential_counterarguments`, `confidence_score_0_to_1`, and `final_answer`. The `final_answer` must be one of: A, B, C, or D."
    return base + constraint


def create_prompt_level_3(example: Example) -> str:
    """Level 3 - Micromanager (Structural Rigidity)."""
    base = chain_of_thought_prompt(example)
    constraint = "\n\nIMPORTANT: Your reasoning must consist of exactly three bullet points. Each bullet point must be no longer than 20 words. You must not use the word 'because' or 'therefore'."
    return base + constraint


def create_prompt_level_4(example: Example) -> str:
    """Level 4 - Pythonic Weaver (Context Switching)."""
    base = chain_of_thought_prompt(example)
    constraint = "\n\nIMPORTANT: Write your reasoning as valid Python code. You must use python comments (`#`) for the logic, but every two lines of comments, you must insert a valid `print()` statement that summarizes the current logical state of the problem. At the end, include a comment with your final answer in the format: # Final answer: (A/B/C/D)"
    return base + constraint


def create_prompt_level_5(example: Example) -> str:
    """Level 5 - Oulipo (Maximum Cognitive Load)."""
    base = chain_of_thought_prompt(example)
    constraint = "\n\nIMPORTANT: Provide your reasoning step-by-step, but **do not use the letter 'e'** in your entire reasoning chain. If you use the letter 'e', you fail. At the end, clearly state your final answer in the format: The correct answer is (A/B/C/D)"
    return base + constraint


PROMPT_FUNCTIONS = [
    create_prompt_level_1,
    create_prompt_level_2,
    create_prompt_level_3,
    create_prompt_level_4,
    create_prompt_level_5,
]


def parse_answer(response: str, constraint_level: int) -> Optional[str]:
    """Parse A/B/C/D answer from response based on constraint level."""
    # Level 2: JSON response
    if constraint_level == 2:
        try:
            # Try to extract JSON
            json_match = re.search(r'\{[^{}]*"final_answer"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                answer = data.get('final_answer', '').upper().strip()
                if answer in LETTER_TO_INDEX:
                    return answer
        except (json.JSONDecodeError, KeyError):
            pass
    
    # Level 4: Python code response
    if constraint_level == 4:
        # Look for final answer comment
        final_answer_match = re.search(r'#\s*Final\s+answer:\s*\(?([A-D])\)?', response, re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1).upper()
    
    # Default: Use regex patterns from example.py
    patterns = [
        r'answer is \((.)\)',
        r'Answer: \((.)\)',
        r'answer: \((.)\)',
        r'answer \((.)\)',
        r'correct answer is \((.)\)',
        r'Correct answer is \((.)\)',
        r'final answer is \((.)\)',
        r'Final answer is \((.)\)',
        r'\(([A-D])\)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match and match.group(1).upper() in LETTER_TO_INDEX:
            return match.group(1).upper()
    
    return None


def create_batch_requests(examples: List[Example]) -> List[Dict]:
    """Create batch API requests in cyclic order: Level 1 (all Q, rep 1) → Level 2 (all Q, rep 1) → ... → Level 5 (all Q, rep 5)."""
    requests = []
    
    # Pre-compute cacheable prefix for Level 1 if examples file exists
    cacheable_prefix = None
    if os.path.exists("prompts/chain_of_thought_examples.json"):
        examples_prefix = "Here are some example questions from experts. An explanation is given before the final answer. Answer the final question yourself, giving your reasoning beforehand.\n"
        json_data = load_from_json("prompts/chain_of_thought_examples.json")
        cacheable_prefix = examples_prefix + generate_prompt_from_examples(json_data, with_explanations=True)
    
    # Processing order: For each repetition, cycle through all constraint levels
    for repetition in range(1, REPETITIONS + 1):
        for constraint_level in range(1, CONSTRAINT_LEVELS + 1):
            for question_id, example in enumerate(examples):
                prompt_func = PROMPT_FUNCTIONS[constraint_level - 1]
                prompt = prompt_func(example)
                
                # Create message for Anthropic API
                custom_id = f"q{question_id}_c{constraint_level}_r{repetition}"
                
                # Prepare request parameters
                params = {
                    "model": MODEL,
                    "max_tokens": 4096,
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 64000,
                    },
                    "interleaved_scratchpad": True,
                    "effort": "high",
                }
                
                # Add prompt caching for Level 1 (chain of thought with examples)
                if constraint_level == 1 and cacheable_prefix:
                    # Split prompt into cacheable and non-cacheable parts
                    question_part = f"Question: {example.question}\nChoices:\n(A) {example.choice1}\n(B) {example.choice2}\n(C) {example.choice3}\n(D) {example.choice4}\nGive step by step reasoning before you answer, and when you're ready to answer, please use the format \"The correct answer is (insert answer here)\":\n"
                    
                    params["prompt_cache"] = {
                        "type": "ephemeral",
                    }
                    messages = [
                        {"role": "user", "content": cacheable_prefix},
                        {"role": "user", "content": question_part}
                    ]
                else:
                    messages = [{"role": "user", "content": prompt}]
                
                request = {
                    "custom_id": custom_id,
                    "params": params,
                    "messages": messages,
                }
                
                requests.append(request)
    
    logfire.info(f"Created {len(requests)} batch requests")
    return requests


def submit_batch(requests: List[Dict]) -> str:
    """Submit batch to Anthropic API and return batch ID."""
    logfire.info(f"Submitting batch with {len(requests)} requests")
    
    batch = anthropic.beta.messages.Batches.create(
        requests=requests
    )
    
    logfire.info(f"Batch submitted with ID: {batch.id}, status: {batch.processing_status}")
    return batch.id


def wait_for_batch_completion(batch_id: str) -> List:
    """Wait for batch to complete and return results."""
    logfire.info(f"Waiting for batch {batch_id} to complete")
    
    import time
    while True:
        batch = anthropic.beta.messages.Batches.retrieve(batch_id)
        logfire.info(f"Batch status: {batch.processing_status}")
        
        if batch.processing_status == "completed":
            break
        
        if batch.processing_status in ["cancelled", "expired", "failed"]:
            raise RuntimeError(f"Batch {batch_id} failed with status: {batch.processing_status}")
        
        time.sleep(30)  # Wait 30 seconds before checking again
    
    # Retrieve results - the API returns a list of results
    results = anthropic.beta.messages.Batches.retrieve_results(batch_id)
    logfire.info(f"Batch completed, retrieved {len(results)} results")
    return results


def process_batch_results(results: List, examples: List[Example]) -> List[Dict]:
    """Process batch results and extract answers."""
    processed = []
    
    for result in tqdm(results, desc="Processing results"):
        # Extract custom_id from result
        custom_id = getattr(result, 'custom_id', None) or getattr(result, 'customId', None)
        if not custom_id:
            logfire.warn(f"Result missing custom_id: {result}")
            continue
        
        # Parse custom_id: q{question_id}_c{constraint_level}_r{repetition}
        match = re.match(r'q(\d+)_c(\d+)_r(\d+)', str(custom_id))
        if not match:
            logfire.warn(f"Could not parse custom_id: {custom_id}")
            continue
        
        question_id = int(match.group(1))
        constraint_level = int(match.group(2))
        repetition = int(match.group(3))
        
        # Extract response text - handle different result structures
        response_text = ""
        if hasattr(result, 'output'):
            output = result.output
            if hasattr(output, 'text'):
                response_text = output.text
            elif isinstance(output, list) and len(output) > 0:
                # Messages API returns list of content blocks
                for block in output:
                    if hasattr(block, 'text'):
                        response_text += block.text
                    elif isinstance(block, dict) and 'text' in block:
                        response_text += block['text']
            elif isinstance(output, str):
                response_text = output
        elif hasattr(result, 'text'):
            response_text = result.text
        elif isinstance(result, dict):
            # Handle dict-based results
            if 'output' in result:
                output = result['output']
                if isinstance(output, list):
                    for block in output:
                        if isinstance(block, dict) and 'text' in block:
                            response_text += block['text']
                elif isinstance(output, str):
                    response_text = output
        
        # Parse answer
        parsed_answer = parse_answer(response_text, constraint_level)
        example = examples[question_id]
        is_correct = parsed_answer and LETTER_TO_INDEX.get(parsed_answer) == example.correct_index
        
        processed.append({
            'question_id': question_id,
            'constraint_level': constraint_level,
            'repetition': repetition,
            'parsed_answer': parsed_answer,
            'correct': is_correct,
            'raw_response': response_text,
            'expected_answer': list(LETTER_TO_INDEX.keys())[example.correct_index],
        })
    
    logfire.info(f"Processed {len(processed)} results")
    return processed


class AccuracyEvaluator(Evaluator[str, str]):
    """Custom evaluator for accuracy calculation."""
    
    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        """Return 1.0 if correct, 0.0 if incorrect."""
        if ctx.output == ctx.expected_output:
            return 1.0
        return 0.0


def create_pydantic_evals_dataset(processed_results: List[Dict], examples: List[Example]) -> Tuple[Dataset, Dict[int, Dict[str, int]]]:
    """Create Pydantic Evals dataset from processed results and return aggregated stats."""
    cases = []
    aggregated_stats = {}
    
    # Group results by question and constraint level
    results_by_case = {}
    for result in processed_results:
        key = (result['question_id'], result['constraint_level'])
        if key not in results_by_case:
            results_by_case[key] = []
        results_by_case[key].append(result)
    
    # Create cases - one per question-constraint combination
    for (question_id, constraint_level), results_list in results_by_case.items():
        example = examples[question_id]
        expected_answer = list(LETTER_TO_INDEX.keys())[example.correct_index]
        
        # Aggregate results across repetitions - use most common answer
        answers = [r['parsed_answer'] for r in results_list if r['parsed_answer']]
        if answers:
            from collections import Counter
            most_common_answer = Counter(answers).most_common(1)[0][0]
        else:
            most_common_answer = None
        
        correct_count = sum(1 for r in results_list if r['correct'])
        total_count = len(results_list)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        case = Case(
            name=f"q{question_id}_c{constraint_level}",
            inputs={
                'question': example.question,
                'choices': [example.choice1, example.choice2, example.choice3, example.choice4],
                'constraint_level': constraint_level,
            },
            expected_output=expected_answer,
            metadata={
                'question_id': question_id,
                'constraint_level': constraint_level,
                'accuracy': accuracy,
                'correct_count': correct_count,
                'total_count': total_count,
                'most_common_answer': most_common_answer,
            }
        )
        cases.append(case)
        
        # Store aggregated stats
        if constraint_level not in aggregated_stats:
            aggregated_stats[constraint_level] = {'correct': 0, 'total': 0}
        aggregated_stats[constraint_level]['correct'] += correct_count
        aggregated_stats[constraint_level]['total'] += total_count
    
    dataset = Dataset(cases=cases)
    dataset.add_evaluator(AccuracyEvaluator())
    
    return dataset, aggregated_stats


def analyze_results(processed_results: List[Dict]) -> Dict:
    """Calculate accuracy per constraint level."""
    accuracy_by_level = {}
    
    for constraint_level in range(1, CONSTRAINT_LEVELS + 1):
        level_results = [r for r in processed_results if r['constraint_level'] == constraint_level]
        if level_results:
            correct = sum(1 for r in level_results if r['correct'])
            total = len(level_results)
            accuracy = correct / total if total > 0 else 0.0
            accuracy_by_level[constraint_level] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
            }
    
    return accuracy_by_level


def save_results_csv(processed_results: List[Dict], filename: str):
    """Save processed results to CSV."""
    os.makedirs("logs", exist_ok=True)
    filepath = os.path.join("logs", filename)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['question_id', 'constraint_level', 'repetition', 'parsed_answer', 'correct', 'expected_answer', 'raw_response'])
        
        for result in processed_results:
            writer.writerow([
                result['question_id'],
                result['constraint_level'],
                result['repetition'],
                result['parsed_answer'],
                result['correct'],
                result['expected_answer'],
                result['raw_response'][:1000],  # Truncate long responses
            ])
    
    logfire.info(f"Saved results to {filepath}")


def main():
    """Main experiment execution."""
    logfire.info("Starting Bureaucratic Jam Hypothesis Experiment")
    
    # 1. Load dataset
    examples = load_gpqa_dataset()
    if len(examples) < NUM_QUESTIONS:
        logfire.warn(f"Only loaded {len(examples)} examples, expected {NUM_QUESTIONS}")
        examples = examples[:NUM_QUESTIONS]
    else:
        examples = examples[:NUM_QUESTIONS]
    
    # 2. Create batch requests
    requests = create_batch_requests(examples)
    
    # 3. Submit batch
    batch_id = submit_batch(requests)
    
    # 4. Wait for completion and retrieve results
    results = wait_for_batch_completion(batch_id)
    
    # 5. Process results
    processed_results = process_batch_results(results, examples)
    
    # 6. Analyze results
    accuracy_by_level = analyze_results(processed_results)
    
    logfire.info("Accuracy by constraint level:")
    for level, stats in accuracy_by_level.items():
        logfire.info(f"Level {level}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
    
    # 7. Create Pydantic Evals dataset for reporting
    dataset, aggregated_stats = create_pydantic_evals_dataset(processed_results, examples)
    
    # Print aggregated accuracy by constraint level
    logfire.info("Aggregated accuracy by constraint level:")
    for level in sorted(aggregated_stats.keys()):
        stats = aggregated_stats[level]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        logfire.info(f"Level {level}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
    
    # 8. Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_results_csv(processed_results, f"bureaucratic_jam_results_{timestamp}.csv")
    
    logfire.info("Experiment completed")


if __name__ == "__main__":
    main()

