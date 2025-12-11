import sys
from pathlib import Path
from typing import Dict, Optional

from bespokelabs import curator
from datasets import load_dataset
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

load_dotenv()

LEAKAGE_CHECK_PROMPT_TEMPLATE = """
You are a strict guard model checking for data leakage in a personalized response.
The response was written for a user with a specific persona.
Your goal is to determine if the response **explicitly leaks** the persona information.

Persona: {persona}

Response: {response}

**Definitions:**
- **Direct Leakage (BAD):** The response explicitly mentions the user's persona, traits, or demographics. 
  - Examples: "As a software engineer...", "Since you live in New York...", "Given your interest in hiking..."
  - Even if it's polite, if it explicitly states the persona attribute, it is leakage.
- **Personalization (GOOD):** The response adapts the content, tone, or examples to suit the persona *without* explicitly stating the traits.
  - Examples: Using technical jargon for an expert (without saying "since you are an expert"), recommending local hiking trails (without saying "since you live here").

**Task:**
Does the response contain **Direct Leakage** of the persona?
Return a JSON object with:
- leakage_detected: boolean (True if direct leakage is found, False otherwise)
- rationale: string (Explain why it is or isn't leakage, citing specific phrases if applicable)
"""

class LeakageResult(BaseModel):
    leakage_detected: bool = Field(
        description="True if the response explicitly mentions or leaks the persona traits. False if it is merely personalized."
    )
    rationale: str = Field(description="Explanation for the decision.")

class LeakageChecker(curator.LLM):
    response_format = LeakageResult

    def prompt(self, input: dict) -> str:
        return LEAKAGE_CHECK_PROMPT_TEMPLATE.format(
            persona=input["ground_truth_persona"],
            response=input["personalized_response"],
        )

    def parse(self, input: dict, response: LeakageResult) -> Dict:
        # If leakage is detected, the final reward becomes 0.
        # If no leakage, the final reward stays as the original reward (which is 1 for these cases).
        # Note: We only run this on rows where reward == 1.
        
        final_reward = 0 if response.leakage_detected else input["reward"]
        
        return {
            **input,
            "leakage_detected": response.leakage_detected,
            "leakage_rationale": response.rationale,
            "final_reward": final_reward
        }

if __name__ == "__main__":
    from config import (
        DIMENSION_NAME,
        RESPONSE_GEN_MODEL,
        JUDGE_MODEL,
        BACKEND,
        BACKEND_PARAMS,
    )

    print(f"Checking for leakage for dimension: {DIMENSION_NAME}")

    # Load dataset from perceivability step
    input_dataset_name = config.get_dataset_name_with_model(
        "perceivability", RESPONSE_GEN_MODEL
    )
    print(f"Loading dataset from: {input_dataset_name}")
    dataset = load_dataset(input_dataset_name, split="train")

    # Filter for rows where reward is 1 (correctly identified persona)
    # We only need to check leakage if the persona was perceivable.
    # If reward is 0, leakage check is less critical for the score (it's already 0), 
    # but technically a wrong answer could still leak. 
    # However, the user specified: "see all the rows with reward 1"
    
    # We will process ALL rows to keep the dataset complete, but only run the LLM check on reward=1.
    # For reward=0, we set leakage_detected=False (assumption) and final_reward=0.
    
    print(f"Running leakage check using model {JUDGE_MODEL}...")

    checker_kwargs = {"model_name": JUDGE_MODEL}
    if BACKEND is not None:
        checker_kwargs["backend"] = BACKEND
    if BACKEND_PARAMS is not None:
        checker_kwargs["backend_params"] = BACKEND_PARAMS

    checker = LeakageChecker(**checker_kwargs)
    
    # Separate rows that need checking
    rows_to_check = [row for row in dataset if row["reward"] == 1]
    rows_no_check = [row for row in dataset if row["reward"] != 1]
    
    print(f"Checking {len(rows_to_check)} rows where reward == 1...")
    
    # Process rows to check
    if rows_to_check:
        # checker(rows_to_check) returns a CuratorResponse
        # CuratorResponse.dataset is the HF Dataset
        checked_rows = checker(rows_to_check).dataset.to_list()
    else:
        checked_rows = []

    # Process rows that didn't need checking (add default values)
    unchecked_rows = []
    for row in rows_no_check:
        new_row = row.copy()
        new_row["leakage_detected"] = False
        new_row["leakage_rationale"] = "Skipped check (reward != 1)"
        new_row["final_reward"] = row["reward"] # Should be 0
        unchecked_rows.append(new_row)
        
    # Combine and sort to maintain original order if possible, or just combine
    # To maintain order, we could map by ID if available, but simple concatenation is usually fine for HF datasets
    # unless order matters strictly.
    
    all_rows = checked_rows + unchecked_rows
    
    # Convert back to HF dataset
    from datasets import Dataset
    final_dataset = Dataset.from_list(all_rows)

    # Push to HuggingFace Hub
    output_dataset_name = config.get_dataset_name_with_model(
        "leakage_check", RESPONSE_GEN_MODEL
    )
    print(f"Pushing dataset to: {output_dataset_name}")
    final_dataset.push_to_hub(output_dataset_name)
