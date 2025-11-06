import sys
import random
from pathlib import Path
from typing import Dict

from bespokelabs import curator
from datasets import load_dataset
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

load_dotenv()

# TODO: Prompt Template
TEST_PERCEIVABILITY_PROMPT_TEMPLATE = """
You are an unbiased judge that is given a question and a response from a helpful assistant. 
Your job is to test the personalization perceivability level of the response.
To do this, you will be given {num_personas} personas and you will need to infer the persona based on the response.

Response: {response}

{personas}

Return your final answer as a JSON object with the following fields:
- choice: string (the letter of the choice of the persona from the given personas)
  - Output one of the following letters: A, B, C, D, E, F
- rationale: string (the rationale for the choice)
"""


class PerceivabilityTestResult(BaseModel):
    choice: str = Field(
        description="The choice of the persona based on the response. Must be one of: A, B, C, D, E, F."
    )
    rationale: str = Field(description="The rationale for the choice")
    # leakage_detection: bool = Field(
    #     description="Whether the response leaks the persona information of the user and there the judge is able to infer the persona easily"
    # )


class PerceivabilityTestGenerator(curator.LLM):
    response_format = PerceivabilityTestResult

    def prompt(self, input: dict) -> str:
        ground_truth_persona = input["ground_truth_persona"]
        distractor_personas = input["distractor_personas"]

        # Combine all personas and shuffle to avoid ordering bias
        all_personas = distractor_personas + [ground_truth_persona]

        # Create a shuffled list with indices to track ground truth position
        persona_indices = list(range(len(all_personas)))
        random.shuffle(persona_indices)

        # Find which position the ground truth ended up in
        ground_truth_original_index = len(
            distractor_personas
        )  # Last position before shuffle
        ground_truth_position = persona_indices.index(ground_truth_original_index)

        # Map position to letter (A, B, C, ...)
        correct_choice_letter = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[ground_truth_position]

        # Store the correct choice in input for later use in parse()
        input["correct_choice"] = correct_choice_letter

        # Build shuffled personas list
        shuffled_personas = [all_personas[i] for i in persona_indices]
        personas = "\n".join(
            [
                f"{letter}: {persona}"
                for letter, persona in zip(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    shuffled_personas,
                )
            ]
        )

        return TEST_PERCEIVABILITY_PROMPT_TEMPLATE.format(
            num_personas=input["num_distractors"] + 1,
            response=input["personalized_response"],
            personas=personas,
        )

    def parse(self, input: dict, response: PerceivabilityTestResult) -> Dict:
        choice = response.choice
        assert choice in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Get the correct choice that was stored during prompt generation
        correct_choice = input["correct_choice"]
        reward = 1 if choice == correct_choice else 0

        return {
            **input,
            "judge_choice": response.choice,
            "judge_rationale": response.rationale,
            "correct_choice": correct_choice,
            "reward": reward,
        }


if __name__ == "__main__":
    from config import (
        DIMENSION_NAME,
        RESPONSE_GEN_MODEL,
        JUDGE_MODEL,
        BACKEND,
        BACKEND_PARAMS,
    )

    print(f"Testing perceivability for dimension: {DIMENSION_NAME}")

    # Load dataset from previous step
    input_dataset_name = config.get_dataset_name_with_model(
        "responses", RESPONSE_GEN_MODEL
    )
    print(f"Loading dataset from: {input_dataset_name}")
    dataset = load_dataset(input_dataset_name, split="train")

    # Run perceivability test
    backend_info = f" with backend {BACKEND}" if BACKEND else ""
    print(
        f"Running perceivability test using judge model {JUDGE_MODEL}{backend_info}..."
    )
    print(f"Evaluating responses from: {RESPONSE_GEN_MODEL}")

    generator_kwargs = {"model_name": JUDGE_MODEL}
    if BACKEND is not None:
        generator_kwargs["backend"] = BACKEND
    if BACKEND_PARAMS is not None:
        generator_kwargs["backend_params"] = BACKEND_PARAMS

    perceivability_test_generator = PerceivabilityTestGenerator(
        **generator_kwargs,
    )
    dataset_with_perceivability_test = perceivability_test_generator(dataset)

    # Push to HuggingFace Hub (keyed by response model, not judge model)
    output_dataset_name = config.get_dataset_name_with_model(
        "perceivability", RESPONSE_GEN_MODEL
    )
    print(f"Pushing dataset to: {output_dataset_name}")
    dataset_with_perceivability_test.dataset.push_to_hub(output_dataset_name)
