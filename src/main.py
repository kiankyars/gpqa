import json, sys, os, random, re
from collections import namedtuple
from datetime import datetime

import logfire
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from datasets import load_dataset
from tqdm import tqdm

# Initialize Logfire and instrument Pydantic AI for token tracking
logfire.configure()
logfire.instrument_pydantic_ai()

# Example named tuple
Example = namedtuple('Example', ['question', 'choice1', 'choice2', 'choice3', 'choice4', 'correct_index'])

# base 2 config file over the six constraints
PROMPTS = 0b111000
REPETITIONS = 5
MODEL = "claude-opus-4-5-20251101"


def load_gpqa_dataset():
    """Load GPQA Diamond dataset from Hugging Face."""
    with logfire.span("Loading GPQA Diamond dataset"):
        # everything is just called train
        dataset = load_dataset("idavidrein/gpqa", "gpqa_diamond", cache_dir="data")['train']
        
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
            correct_answer = shuffled_choices.index(item['Correct Answer'])
            
            # saving this example
            examples.append(Example(
                question=question,
                choice1=shuffled_choices[0],
                choice2=shuffled_choices[1],
                choice3=shuffled_choices[2],
                choice4=shuffled_choices[3],
                correct_index=correct_answer
            ))
        
        logfire.info(f"Loaded {len(examples)} examples")
        # return list of all the examples which are themselves Example types that inherits and is a child of the namedtaple class
        return examples


def create_prompt_0(example):
    """Baseline"""
    return f"""Answer the following multiple choice question. The last line of your response should be of the following format: 'solution: $LETTER' (without quotes) where LETTER is one of ABCD.

{example.question}

A) {example.choice1}

B) {example.choice2}

C) {example.choice3}

D) {example.choice4}"""


def create_prompt_1(example):
    """Strict JSON"""
    base = create_prompt_0(example)
    constraint = "\n\nIMPORTANT: You must output valid JSON only. The JSON must contain exactly these five keys: `initial_intuition`, `step_by_step_logic`, `potential_counterarguments`, `confidence_score_0_to_1`, and `solution`."
    return base + constraint


def create_prompt_2(example):
    """Structural Rigidity"""
    base = create_prompt_0(example)
    constraint = "\n\nIMPORTANT: Your reasoning must consist of exactly three bullet points. Each bullet point must be no longer than 20 words. You must not use the word 'because' or 'therefore'."
    return base + constraint


def create_prompt_3(example):
    """Python"""
    base = create_prompt_0(example)
    constraint = "\n\nIMPORTANT: You must write your reasoning as valid Python code."
    return base + constraint


def create_prompt_4(example):
    """Oulipo"""
    base = create_prompt_0(example)
    constraint = "\n\nIMPORTANT: You must not use the letter 'e' in your entire reasoning chain."
    return base + constraint


def create_prompt_5(example):
    """The 16 English tokens with the largest l2 norm token embeddings in gpt oss"""
    base = create_prompt_0(example)
    constraint = "\n\nIMPORTANT: You must not use the following words in your reasoning: accordingly, code, ocode, The, settings, Moreover, description, Let's, This, core, utilizes, revolves, Here's, possibly, logic, thereby."
    return base + constraint


PROMPT_FUNCTIONS = [
    create_prompt_0,
    create_prompt_1,
    create_prompt_2,
    create_prompt_3,
    create_prompt_4,
    create_prompt_5,
]

def create_smoke_test_request(example, prompts_mask):
    """For testing purposes."""
    examples = []
    enabled_prompts = [i for i in range(len(PROMPT_FUNCTIONS)) if (prompts_mask >> i) & 1]
    for prompt in enabled_prompts:
        prompt_text = PROMPT_FUNCTIONS[prompt](example)
        examples.append(Request({
            "custom_id": f"smoke_test_q0_p{prompt}_r0",
            "params": MessageCreateParamsNonStreaming({
                "model": MODEL,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt_text}],
            }),
        }))
    return examples

def create_batch_requests(examples, prompts_mask):
    """Create batch API requests in cyclic order."""
    requests = []
    # Identify which levels are enabled via bitmask
    enabled_prompts = [i for i in range(len(PROMPT_FUNCTIONS)) if (prompts_mask >> i) & 1]
    # print(type(enabled_prompts[0]))
    for repetition in range(REPETITIONS):
        for prompt in enabled_prompts:
            for question_id, example in enumerate(examples):
                prompt_text = PROMPT_FUNCTIONS[prompt](example).strip()
                custom_id = f"q{question_id}_p{prompt}_r{repetition}"
                requests.append(Request({
                    "custom_id": custom_id,
                    "params": MessageCreateParamsNonStreaming({
                        "model": MODEL,
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": 64000
                        },
                        "output_config": {
                            "effort": "high"
                        },
                        "max_tokens": 64000,
                        "messages": [{"role": "user", "content": prompt_text}],
                    }),
                }))
    with logfire.span(f"Created {len(requests)} batch requests"):
        return requests


def submit_batch(requests, client):
    """Submit batch to Anthropic API."""
    with logfire.span(f"Submitting batch with {len(requests)} requests"):
        batch = client.messages.batches.create(
            requests=requests,
            extra_headers={
                "anthropic-beta": "interleaved-thinking-2025-05-14,effort-2025-11-24"
            }
            )
        logfire.info(f"Batch ID: {batch.id}, status: {batch.processing_status}")
        return batch.id


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


def process_batch_results(batch_id, examples, client):
    """Process batch results and extract answers."""
    processed = []
    for result in tqdm(client.messages.batches.results(batch_id), desc="Processing results"):
        match = re.match(r'q(\d+)_p(\d+)_r(\d+)', result.custom_id)
        # print(match, result.custom_id, result)
        question_id = int(match.group(1))
        prompt = int(match.group(2))
        repetition = int(match.group(3))
        response_text = next(block.text for block in result.result.message.content if block.type == "text")
        extracted_letter = extract_answer_from_response(response_text, prompt)
        # this is the index of the correct answer
        correct_answer = examples[question_id].correct_index
        # this is the letter of the correct answer
        correct_letter = 'ABCD'[correct_answer]
        # checking if the letters are the same
        score = 1.0 if extracted_letter == correct_letter else 0.0
        
        processed.append({
            'question_id': question_id,
            'prompt': prompt,
            'repetition': repetition,
            'extracted_letter': extracted_letter,
            'score': score,
            'correct_answer': correct_answer,
            'input_tokens': result.result.message.usage.input_tokens,
            'output_tokens': result.result.message.usage.output_tokens,
            'response_text': response_text,
        })
    return processed


def save_results_jsonl(processed_results, filename):
    """Save processed results."""
    filepath = os.path.join("data", filename)
    with logfire.span(f"Saving results to {filepath}"):
        with open(filepath, 'w') as outfile:
            for entry in processed_results:
                json.dump(entry, outfile)
                # Adds a newline character to create the JSONL format
                outfile.write('\n')


def main():
    with logfire.span(f"Starting with model {MODEL}, prompts {PROMPTS:b}, repetitions {REPETITIONS}"):
        examples = load_gpqa_dataset()
        client = Anthropic()
        if sys.argv[1] == "submit":
            if sys.argv[2] == "smoke":
                requests = create_smoke_test_request(examples[0], PROMPTS)
            elif sys.argv[2] == "full":
                requests = create_batch_requests(examples, PROMPTS)
            submit_batch(requests, client)
        else:
            batch_id = sys.argv[1]
            processed_results = process_batch_results(batch_id, examples, client)
            # Save
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_results_jsonl(processed_results, f"results_{timestamp}_{PROMPTS}.jsonl")


if __name__ == "__main__":
    main()