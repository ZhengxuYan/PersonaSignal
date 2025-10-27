from typing import Dict, List

from bespokelabs import curator
from pydantic import BaseModel, Field

# TODO: Prompt Template
TEST_PERCEIVABILITY_PROMPT_TEMPLATE = """
You are a helpful assistant that is given a question and a persona.
You need to test the perceivability of the persona based on the question.

Question: {question}
Persona: {persona}

Return your final answer as a JSON object with the following fields:
- perceivability: string (the perceivability of the persona based on the question)
- rationale: string (the rationale for the perceivability)
- confidence: float (the confidence in the perceivability)
"""


class PerceivabilityTestResult(BaseModel):
  perceivability: str = Field(
      description="The perceivability of the persona based on the question"
  )
  rationale: str = Field(
      description="The rationale for the perceivability"
  )
  confidence: float = Field(
      description="The confidence in the perceivability"
  )


class PerceivabilityTestGenerator(curator.LLM):
  response_format = PerceivabilityTestResult

  def prompt(self, input: dict) -> str:
    return TEST_PERCEIVABILITY_PROMPT_TEMPLATE.format(
        question=input["question"],
        persona=input["persona"]
    )

  def parse(self, input: dict, response: PerceivabilityTestResult) -> Dict:
    return {
        **input,
        "perceivability": response.perceivability,
        "rationale": response.rationale,
        "confidence": response.confidence
    }


MODEL_NAME = "gpt-4o-mini"
dataset = Dataset.from_pandas(pd.read_csv(
    "data_with_personalized_response.csv"))
perceivability_test_generator = PerceivabilityTestGenerator(
    model_name=MODEL_NAME)
dataset_with_perceivability_test = perceivability_test_generator(dataset)
print(dataset_with_perceivability_test.dataset)

# TODO: Random Order to avoid ordering bias
