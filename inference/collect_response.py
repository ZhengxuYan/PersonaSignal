import sys
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

PERSONALIZED_RESPONSE_SYSTEMPROMPT_TEMPLATE = """
You are a helpful assistant responding to a user with the following characteristics: {persona}

**Critical Instructions:**
1. Adapt your response to naturally align with this user's background, preferences, and context
2. NEVER explicitly mention, reference, or describe the user's persona traits in your response
3. NEVER use phrases like "given your [trait]", "since you are [description]", or "as someone who [characteristic]"
4. Let the personalization emerge implicitly through:
   - The reasoning approach and depth you choose
   - The examples, priorities, or framing you emphasize
   - The assumptions you make about context
   - The level of detail and abstraction
   - The order and focus of your points

Your goal: Provide a response that would naturally suit this user without making the persona obvious or detectable through explicit mentions.
"""


class PersonalizedResponse(BaseModel):
    personalized_response: str = Field(
        description="The personalized response to the question"
    )


class PersonalizedResponseGenerator(curator.LLM):
    response_format = PersonalizedResponse

    def prompt(self, input: dict) -> str:
        system_prompt = PERSONALIZED_RESPONSE_SYSTEMPROMPT_TEMPLATE.format(
            persona=input["ground_truth_persona"]
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input["question"]},
        ]

    def parse(self, input: dict, response: PersonalizedResponse) -> Dict:
        return {**input, "personalized_response": response.personalized_response}


if __name__ == "__main__":
    from config import (
        DIMENSION_NAME,
        RESPONSE_GEN_MODEL,
        QUESTION_GEN_MODEL,
        BACKEND,
    )

    print(f"Generating personalized responses for dimension: {DIMENSION_NAME}")

    # Load dataset from previous step
    input_dataset_name = config.get_dataset_name_with_model(
        "questions", QUESTION_GEN_MODEL
    )
    print(f"Loading dataset from: {input_dataset_name}")
    dataset = load_dataset(input_dataset_name, split="train")

    # Generate personalized responses
    print(
        f"Generating personalized responses using {RESPONSE_GEN_MODEL} with backend {BACKEND}..."
    )
    personalized_response_generator = PersonalizedResponseGenerator(
        model_name=RESPONSE_GEN_MODEL, backend=BACKEND
    )
    dataset_with_personalized_response = personalized_response_generator(dataset)

    # Push to HuggingFace Hub
    output_dataset_name = config.get_dataset_name_with_model(
        "responses", RESPONSE_GEN_MODEL
    )
    print(f"Pushing dataset to: {output_dataset_name}")
    dataset_with_personalized_response.dataset.push_to_hub(output_dataset_name)
