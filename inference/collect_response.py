from typing import Dict, List

import pandas as pd
from bespokelabs import curator
from datasets import Dataset, load_dataset
from pydantic import BaseModel, Field

# TODO: Prompt Template
PERSONALIZED_RESPONSE_SYSTEMPROMPT_TEMPLATE = """
You are a helpful assistant. The user you are helping is {persona}. Do not directlyleak persona information of the user in your response.
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
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": input["question"]}]

  def parse(self, input: dict, response: PersonalizedResponse) -> Dict:
    return {
        **input,
        "personalized_response": response.personalized_response
    }


MODEL_NAME = "gpt-4o-mini"
dataset = load_dataset("RZ412/PersonaSignal-PersonaQuestions", split="train")
personalized_response_generator = PersonalizedResponseGenerator(
    model_name=MODEL_NAME)

dataset_with_personalized_response = personalized_response_generator(dataset)

dataset_with_personalized_response.dataset.push_to_hub(
    "RZ412/PersonaSignal-PersonalizedResponse")
