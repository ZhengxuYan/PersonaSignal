from typing import Dict, List

import pandas as pd
from bespokelabs import curator
from datasets import Dataset
from pydantic import BaseModel, Field

# TODO: Prompt Template
PERSONALIZED_RESPONSE_PROMPT_TEMPLATE = """
You are a helpful assistant that is given a question and a dimension value.
You need to generate a personalized response to the question based on the dimension value.

Question: {question}
Dimension Name: {dimension_name}
Dimension Value: {dimension_value}

"""


class PersonalizedResponse(BaseModel):
  personalized_response: str = Field(
      description="The personalized response to the question"
  )


class PersonalizedResponseGenerator(curator.LLM):
  response_format = PersonalizedResponse

  def prompt(self, input: dict) -> str:
    return PERSONALIZED_RESPONSE_PROMPT_TEMPLATE.format(
        question=input["question"],
        dimension_name=input["dimension_name"],
        dimension_value=input["dimension_value"]
    )

  def parse(self, input: dict, response: PersonalizedResponse) -> Dict:
    return {
        **input,
        "personalized_response": response.personalized_response
    }


MODEL_NAME = "gpt-4o-mini"
dataset = Dataset.from_pandas(pd.read_csv("data_with_personas.csv"))
personalized_response_generator = PersonalizedResponseGenerator(
    model_name=MODEL_NAME)

dataset_with_personalized_response = personalized_response_generator(dataset)

print(dataset_with_personalized_response.dataset)
