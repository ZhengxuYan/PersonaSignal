import sys
import random
import asyncio
from pathlib import Path
from typing import Dict, List

from bespokelabs import curator
from datasets import load_dataset, Dataset
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import pandas as pd

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

load_dotenv()

# Configuration
DATASET_A_NAME = "JasonYan777/PersonaSignal-All-Leakage_Check-Meta-Llama-3.1-8B-Instruct-Turbo"
DATASET_B_NAME = "JasonYan777/PersonaSignal-All-Leakage_Check-DPO-Tinker"
MODEL_A_ALIAS = "Meta Llama 3.1 8B Instruct Turbo"
MODEL_B_ALIAS = "DPO Tinker"

JUDGE_PROMPT_TEMPLATE = """
You are an impartial judge evaluating the quality of personalized responses from two AI assistants.

**User Persona:**
{persona}

**Question:**
{question}

**Response A:**
{response_a}

**Response B:**
{response_b}

**Evaluation Criteria:**
1. **Personalization**: Does the response implicitly adapt to the persona's background, values, and knowledge level without being explicit (no "As a [persona]...")?
2. **Usefulness**: Is the answer helpful and accurate for the specific question?
3. **Tone/Style**: Is the tone appropriate for the persona (e.g., professional for experts, simple for novices)?

Which response is better specifically for this user persona?

Return your decision as a JSON object with:
- `choice`: "A" or "B"
- `rationale`: A brief explanation of why the chosen response is better.
"""

class ComparisonResult(BaseModel):
    choice: str = Field(description="The chosen response, either 'A' or 'B'.")
    rationale: str = Field(description="Explanation for the choice.")

class ComparisonGenerator(curator.LLM):
    response_format = ComparisonResult

    def prompt(self, input: dict) -> str:
        # Determine order for this sample (randomize to avoid position bias)
        # We stored the randomization decision in the input during preparation
        if input["is_swapped"]:
             response_a = input["response_b_content"]
             response_b = input["response_a_content"]
        else:
             response_a = input["response_a_content"]
             response_b = input["response_b_content"]

        return JUDGE_PROMPT_TEMPLATE.format(
            persona=input["ground_truth_persona"],
            question=input["question"],
            response_a=response_a,
            response_b=response_b
        )

    def parse(self, input: dict, response: ComparisonResult) -> Dict:
        # Map choice back to model names
        choice = response.choice.strip().upper()
        
        if input["is_swapped"]:
            # If swapped: A is Model B, B is Model A
            winner_model = MODEL_B_ALIAS if choice == "A" else MODEL_A_ALIAS
        else:
            # If not swapped: A is Model A, B is Model B
            winner_model = MODEL_A_ALIAS if choice == "A" else MODEL_B_ALIAS

        return {
            **input,
            "judge_choice_raw": choice,
            "judge_rationale": response.rationale,
            "winner_model": winner_model
        }

def load_and_align_datasets():
    print(f"Loading Dataset A: {DATASET_A_NAME}")
    ds_a = load_dataset(DATASET_A_NAME, split="train")
    
    print(f"Loading Dataset B: {DATASET_B_NAME}")
    ds_b = load_dataset(DATASET_B_NAME, split="train")
    
    # Create lookups based on question + persona
    # Using a composite key of (question, persona) to ensure uniqueness
    def create_lookup(dataset):
        lookup = {}
        for row in dataset:
            key = (row["question"], row["ground_truth_persona"])
            if key in lookup:
                print(f"Warning: Duplicate key found for {key}")
            lookup[key] = row
        return lookup

    lookup_a = create_lookup(ds_a)
    lookup_b = create_lookup(ds_b)
    
    # Find common keys
    common_keys = set(lookup_a.keys()) & set(lookup_b.keys())
    print(f"Found {len(common_keys)} common samples between datasets.")
    
    if len(common_keys) < len(lookup_a) or len(common_keys) < len(lookup_b):
        print(f"Warning: Missing matches. A: {len(lookup_a)}, B: {len(lookup_b)}, Common: {len(common_keys)}")
    
    aligned_data = []
    
    for key in common_keys:
        row_a = lookup_a[key]
        row_b = lookup_b[key]
        
        # Prepare input for judge
        # Randomly decide swap
        is_swapped = random.random() > 0.5
        
        item = {
            "question": row_a["question"],
            "ground_truth_persona": row_a["ground_truth_persona"],
            "response_a_content": row_a["personalized_response"], # Model A
            "response_b_content": row_b["personalized_response"], # Model B
            "is_swapped": is_swapped,
            # Keep original metadata if needed
            "dimension_name": row_a.get("dimension_name", ""),
        }
        aligned_data.append(item)
        
    return aligned_data

def main():
    aligned_data = load_and_align_datasets()
    print(f"Prepared {len(aligned_data)} aligned samples for comparison.")
    
    # Configure Backend
    judge_model = config.JUDGE_MODEL
    print(f"Using Judge Model: {judge_model}")
    
    generator_kwargs = {"model_name": judge_model}
    if config.BACKEND is not None:
        generator_kwargs["backend"] = config.BACKEND
    if config.BACKEND_PARAMS is not None:
        generator_kwargs["backend_params"] = config.BACKEND_PARAMS
        
    print("Starting evaluation...")
    comparer = ComparisonGenerator(**generator_kwargs)
    
    # Run batch evaluation
    results = comparer(aligned_data)
    
    # Analysis
    # CuratorResponse usually has a .probs or .response attribute, or we can convert to pandas
    if hasattr(results, 'to_pandas'):
        df = results.to_pandas()
    elif hasattr(results, 'dataset'):
        df = results.dataset.to_pandas()
    else:
        # Fallback or inspection
        try:
            df = pd.DataFrame(results.to_dict())
        except:
             # Last resort, if it is a list
             df = pd.DataFrame(results)

    win_counts = df["winner_model"].value_counts()
    print("\n=== Results ===")
    print(win_counts)
    
    win_rates = df["winner_model"].value_counts(normalize=True)
    print("\n=== Win Rates ===")
    print(win_rates)
    
    # Save detailed results
    output_path = "comparison_results.jsonl"
    df.to_json(output_path, orient="records", lines=True)
    print(f"\nDetailed results saved to {output_path}")

    # Optional: Push to Hub if desired, or just keep local
    
if __name__ == "__main__":
    main()
