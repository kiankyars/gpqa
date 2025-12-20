import csv
import json
import os
import random
import re
from collections import namedtuple
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import logfire
from anthropic import Anthropic
from datasets import load_dataset
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext
from tqdm import tqdm

# Initialize Logfire and instrument Pydantic AI for token tracking
logfire.configure()
logfire.instrument_pydantic_ai()

# Example named tuple
Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

# base 2 config file over the six constraints
PROMPTS = 0b1
REPETITIONS = 5
MODEL = "claude-opus-4-5-20251101"


def load_gpqa_dataset() -> List[Example]:
    """Load GPQA Diamond dataset from Hugging Face."""
    logfire.info("Loading GPQA Diamond dataset")
    # everything is just called train
    dataset = load_dataset("idavidrein/gpqa", "gpqa_diamond")['train']
    
    examples = []
    # random seed for mixing results
    random.seed(0)
    
    for item in dataset:
        # print(item.keys())
        # exit()
        # grabbing the required keys for each question
        # string
        question = item['Question']
        # list of strings
        choices = [
            item['Correct Answer'],
            item['Incorrect Answer 1'],
            item['Incorrect Answer 2'],
            item['Incorrect Answer 3'],
        ]
        # permutation of the list [0,1,2,3]
        permutation = list(range(4))
        # pseudo-random shuffle
        random.shuffle(permutation)
        # shuffled choices with list compression
        shuffled_choices = [choices[i] for i in permutation]
        # grabbing the index of the correct answer
        correct_index = shuffled_choices.index(item['Correct Answer'])
        
        # saving this example
        examples.append(Example(
            question=question,
            choice1=shuffled_choices[0],
            choice2=shuffled_choices[1],
            choice3=shuffled_choices[2],
            choice4=shuffled_choices[3],
            correct_index=correct_index
        ))
    
    logfire.info(f"Loaded {len(examples)} examples")
    # return list of all the examples which are themselves Example types that inherits and is a child of the namedtaple class
    return examples


def create_prompt_1(example: Example) -> str:
    """Baseline"""
    return f"""Answer the following multiple choice question. The last line of your response should be of the following format: 'solution: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{example.question}

A) {example.choice1}

B) {example.choice2}

C) {example.choice3}

D) {example.choice4}"""


def create_prompt_2(example: Example) -> str:
    """Strict JSON"""
    base = f"""Answer the following multiple choice question. The last line of your response should be of the following format: 'solution: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{example.question}

A) {example.choice1}

B) {example.choice2}

C) {example.choice3}

D) {example.choice4}"""
    
    constraint = "\n\nIMPORTANT: You must output valid JSON only. The JSON must contain exactly these five keys: `initial_intuition`, `step_by_step_logic`, `potential_counterarguments`, `confidence_score_0_to_1`, and `solution`."
    return base + constraint


def create_prompt_3(example: Example) -> str:
    """Structural Rigidity"""
    base = f"""Answer the following multiple choice question. The last line of your response should be of the following format: 'solution: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{example.question}

A) {example.choice1}

B) {example.choice2}

C) {example.choice3}

D) {example.choice4}"""
    
    constraint = "\n\nIMPORTANT: Your reasoning must consist of exactly three bullet points. Each bullet point must be no longer than 20 words. You must not use the word 'because' or 'therefore'."
    return base + constraint


def create_prompt_4(example: Example) -> str:
    """Python"""
    base = f"""Answer the following multiple choice question. The last line of your response should be of the following format: 'solution: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{example.question}

A) {example.choice1}

B) {example.choice2}

C) {example.choice3}

D) {example.choice4}"""
    
    constraint = "\n\nIMPORTANT: Write your reasoning as valid Python code."
    return base + constraint


def create_prompt_5(example: Example) -> str:
    """Oulipo"""
    base = f"""Answer the following multiple choice question. The last line of your response should be of the following format: 'solution: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{example.question}

A) {example.choice1}

B) {example.choice2}

C) {example.choice3}

D) {example.choice4}"""
    
    constraint = "\n\nIMPORTANT: Provide your reasoning step-by-step, but **do not use the letter 'e'** in your entire reasoning chain."
    return base + constraint


def create_prompt_6(example: Example) -> str:
    """The 16 English tokens with the largest l2 norm token embeddings in gpt oss"""
    base = f"""Answer the following multiple choice question. The last line of your response should be of the following format: 'solution: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{example.question}

A) {example.choice1}

B) {example.choice2}

C) {example.choice3}

D) {example.choice4}"""
    
    constraint = "\n\nIMPORTANT: Do not use the following words in your reasoning: accordingly, code, ocode, The, settings, Moreover, description, Let's, This, core, utilizes, revolves, Here's, possibly, logic, thereby"
    return base + constraint


PROMPT_FUNCTIONS = [
    create_prompt_1,
    create_prompt_2,
    create_prompt_3,
    create_prompt_4,
    create_prompt_5,
    create_prompt_6,
]


def create_batch_requests(examples: List[Example]) -> List[Dict]:
    """Create batch API requests in cyclic order."""
    requests = []
    
    for repetition in range(1, REPETITIONS + 1):
        for constraint_level in range(1, CONSTRAINT_LEVELS + 1):
            for question_id, example in enumerate(examples):
                prompt = PROMPT_FUNCTIONS[constraint_level - 1](example)
                custom_id = f"q{question_id}_c{constraint_level}_r{repetition}"
                
                requests.append({
                    "custom_id": custom_id,
                    "params": {
                        "model": MODEL,
                        "betas": ["interleaved-thinking-2025-05-14", "effort-2025-11-24"],
                        "thinking_config": {
                            "type": "enabled",
                            "budget_tokens": 64000
                        },
                        "output_config": {
                            "effort": "high"
                        },
                        "max_tokens": 4096,
                    },
                    "messages": [{"role": "user", "content": prompt}],
                })
    
    logfire.info(f"Created {len(requests)} batch requests")
    return requests


def submit_batch(requests: List[Dict]) -> str:
    """Submit batch to Anthropic API."""
    logfire.info(f"Submitting batch with {len(requests)} requests")
    batch = anthropic.beta.messages.Batches.create(requests=requests)
    logfire.info(f"Batch ID: {batch.id}, status: {batch.processing_status}")
    return batch.id


def wait_for_batch_completion(batch_id: str) -> List:
    """Wait for batch to complete and return results."""
    import time
    while True:
        batch = anthropic.beta.messages.Batches.retrieve(batch_id)
        logfire.info(f"Batch status: {batch.processing_status}")
        if batch.processing_status == "completed":
            break
        if batch.processing_status in ["cancelled", "expired", "failed"]:
            raise RuntimeError(f"Batch failed: {batch.processing_status}")
        time.sleep(30)
    
    results = anthropic.beta.messages.Batches.retrieve_results(batch_id)
    logfire.info(f"Retrieved {len(results)} results")
    return results


def process_batch_results(results: List, examples: List[Example]) -> List[Dict]:
    """Process batch results and extract answers."""
    processed = []
    
    for result in tqdm(results, desc="Processing results"):
        custom_id = getattr(result, 'custom_id', None) or getattr(result, 'customId', None)
        if not custom_id:
            continue
        
        match = re.match(r'q(\d+)_c(\d+)_r(\d+)', str(custom_id))
        if not match:
            continue
        
        question_id = int(match.group(1))
        constraint_level = int(match.group(2))
        repetition = int(match.group(3))
        
        # Extract response text
        response_text = ""
        if hasattr(result, 'output'):
            output = result.output
            if hasattr(output, 'text'):
                response_text = output.text
            elif isinstance(output, list):
                for block in output:
                    if hasattr(block, 'text'):
                        response_text += block.text
                    elif isinstance(block, dict) and 'text' in block:
                        response_text += block['text']
            elif isinstance(output, str):
                response_text = output
        
        parsed_answer = parse_answer(response_text, constraint_level)
        example = examples[question_id]
        is_correct = parsed_answer and 'ABCD'[parsed_answer] == example.correct_index
        
        processed.append({
            'question_id': question_id,
            'constraint_level': constraint_level,
            'repetition': repetition,
            'parsed_answer': parsed_answer,
            'correct': is_correct,
            'raw_response': response_text,
            'expected_answer': list(LETTER_TO_INDEX.keys())[example.correct_index],
        })
    
    return processed


class AccuracyEvaluator(Evaluator[str, str]):
    """Custom evaluator for accuracy calculation."""
    
    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        return 1.0 if ctx.output == ctx.expected_output else 0.0


def create_pydantic_evals_dataset(processed_results: List[Dict], examples: List[Example]) -> Tuple[Dataset, Dict[int, Dict[str, int]]]:
    """Create Pydantic Evals dataset from processed results."""
    cases = []
    aggregated_stats = {}
    
    results_by_case = {}
    for result in processed_results:
        key = (result['question_id'], result['constraint_level'])
        if key not in results_by_case:
            results_by_case[key] = []
        results_by_case[key].append(result)
    
    for (question_id, constraint_level), results_list in results_by_case.items():
        example = examples[question_id]
        expected_answer = list(LETTER_TO_INDEX.keys())[example.correct_index]
        
        correct_count = sum(1 for r in results_list if r['correct'])
        total_count = len(results_list)
        
        cases.append(Case(
            name=f"q{question_id}_c{constraint_level}",
            inputs={'question': example.question, 'constraint_level': constraint_level},
            expected_output=expected_answer,
            metadata={'question_id': question_id, 'constraint_level': constraint_level},
        ))
        
        if constraint_level not in aggregated_stats:
            aggregated_stats[constraint_level] = {'correct': 0, 'total': 0}
        aggregated_stats[constraint_level]['correct'] += correct_count
        aggregated_stats[constraint_level]['total'] += total_count
    
    dataset = Dataset(cases=cases)
    dataset.add_evaluator(AccuracyEvaluator())
    return dataset, aggregated_stats


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
                result['raw_response'][:1000],
            ])
    
    logfire.info(f"Saved results to {filepath}")


def main():
    """Main experiment execution."""
    logfire.info(f"Starting with model {MODEL} and constraint levels {PROMPTS} and repetitions {REPETITIONS}")
    
    examples = load_gpqa_dataset()
    assert len(examples) == 198, f"Expected 198 examples, got {len(examples)}"
    exit()
    # Initialize Anthropic client
    anthropic = Anthropic()
    requests = create_batch_requests(examples)
    batch_id = submit_batch(requests)
    results = wait_for_batch_completion(batch_id)
    processed_results = process_batch_results(results, examples)
    
    dataset, aggregated_stats = create_pydantic_evals_dataset(processed_results, examples)
    
    logfire.info("Accuracy by constraint level:")
    for level in sorted(aggregated_stats.keys()):
        stats = aggregated_stats[level]
        accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        logfire.info(f"Level {level}: {accuracy:.3f} ({stats['correct']}/{stats['total']})")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_results_csv(processed_results, f"results_{timestamp}.csv")
    
    logfire.info("Experiment completed")


if __name__ == "__main__":
    main()
