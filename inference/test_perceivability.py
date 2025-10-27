import ast
from enum import Enum
from typing import Dict, List

import pandas as pd
from bespokelabs import curator
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field

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
  rationale: str = Field(
      description="The rationale for the choice"
  )
  # leakage_detection: bool = Field(
  #     description="Whether the response leaks the persona information of the user and there the judge is able to infer the persona easily"
  # )


class PerceivabilityTestGenerator(curator.LLM):
  response_format = PerceivabilityTestResult

  def prompt(self, input: dict) -> str:
    ground_truth_persona = input["ground_truth_persona"]
    distractor_personas = input["distractor_personas"]
    # print(distractor_personas)
    personas = "\n".join([f"{letter}: {persona}" for letter, persona in zip(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ", distractor_personas + [ground_truth_persona])])
    # print(personas)
    # raise Exception("Stop here")
    return TEST_PERCEIVABILITY_PROMPT_TEMPLATE.format(
        num_personas=input["num_distractors"] + 1,
        response=input["personalized_response"],
        personas=personas
    )

  def parse(self, input: dict, response: PerceivabilityTestResult) -> Dict:
    choice = response.choice
    assert choice in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    reward = 1 if choice == "F" else 0
    return {
        **input,
        "judge_choice": response.choice,
        "judge_rationale": response.rationale,
        # "persona_leaked": response.leakage_detection,
        "reward": reward,
    }


MODEL_NAME = "gpt-4o-mini"
dataset = load_dataset(
    "RZ412/PersonaSignal-PersonalizedResponse", split="train")
perceivability_test_generator = PerceivabilityTestGenerator(
    model_name=MODEL_NAME)
dataset_with_perceivability_test = perceivability_test_generator(dataset)

dataset_with_perceivability_test.dataset.push_to_hub(
    "RZ412/PersonaSignal-PersonalizedResponseWithPerceivabilityTest")
