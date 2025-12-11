import random
import sys
from pathlib import Path
from typing import Dict, List

from datasets import Dataset
from bespokelabs import curator
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

load_dotenv()

QUESTION_GENERATOR_PROMPT_TEMPLATE = """
You are designing evaluation data for *third-party perceivability of personalization* in language models.

Context:
In this benchmark, a model (the “assistant”) will generate answers conditioned on a hidden persona. 
A separate model (the “judge”) must infer which persona the assistant was responding to, based only on the text of the answer.  
We want *subtle, non-trivial* personalization effects — not superficial or easily guessable ones.

---

### Your Goal
Given a *persona dimension*, design **questions** that:
1. Naturally elicit **different reasoning patterns, priorities, or content choices** depending on the persona value.  
2. Do **not** make the persona obvious. The difference should be *latent but perceivable* through reasoning or tone — 
   not through explicit phrases, stance keywords, or demographic hints.  
3. Make it **challenging for a judge to guess the persona** — perceivability should depend on nuanced textual behavior 
   (e.g., structure of reasoning, focus, implicit preferences), not direct mentions or vocabulary cues.

---

### Dimension Input:
{dimension}

---

### Output Requirements:

Generate **10 diverse, realistic, and moderately difficult question–rationale–cue triples** where personalization effects are subtle.

Each output should follow this schema:

{{
  "question": "string (a concise, realistic question that could be asked to an assistant)",
  "why_differ": "string (1–2 sentences explaining how the assistant's reasoning or priorities would shift across persona values)",
  "how_subtle": "string (describe how the difference is *implicit*, e.g., through reasoning depth, emphasis, ordering, or abstraction, not through obvious stance words)"
}}

Additional constraints:
- Avoid questions where persona value determines an obvious opinion (“Should I take a risk?” “Do you prefer X or Y?”).
- Avoid explicit lexical leaks (words that directly reveal the persona trait).
- Questions should be answerable in multiple valid ways, each plausible for a different persona value.
- Favor domains where reasoning style or framing subtly shifts: planning, explanation, evaluation, prioritization, reflection.

---

### Example (for reference)

**Dimension:** Risk Attitude – how willing a user is to take uncertain or bold options.

[
  {{
    "question": "I'm considering changing jobs next year. How should I approach the decision process?",
    "why_differ": "A cautious user expects structured risk evaluation and fallback planning; a bold user expects focus on upside and opportunity framing.",
    "how_subtle": "Perceivability arises from reasoning emphasis — cautious responses discuss contingencies, bold ones emphasize growth and self-belief — without explicitly mentioning 'risk'."
  }}
]

---

### Summary of your task
Design {num_questions} question–rationale–cue triples for the given dimension where:
- The difference in ideal answers is *detectable but not explicit*.
- The persona is *hard but not impossible* to infer.
- The questions sound natural for any user, independent of persona wording.

Return your final answer as a JSON list.
"""


class question_list(BaseModel):
    question_list: List[Dict] = Field(
        description="A list of questions with their rationale and how subtle the difference is"
    )


class QuestionGenerator(curator.LLM):
    response_format = question_list

    def prompt(self, dimension: dict) -> str:
        num_questions = dimension.get("num_questions", 10)
        return QUESTION_GENERATOR_PROMPT_TEMPLATE.format(
            dimension=dimension, num_questions=num_questions
        )

    def parse(self, input: dict, response: question_list) -> List[Dict]:
        return [
            {
                "dimension_name": input["name"],
                "dimension_values": input["values"],
                "dimension_description": input.get(
                    "description", input.get("notes", "")
                ),
                "question": q["question"],
                "why_differ": q["why_differ"],
                "how_subtle": q["how_subtle"],
            }
            for q in response.question_list
        ]


PERSONA_GENERATOR_PROMPT_TEMPLATE = """
You are creating persona profiles for testing *perceivability of personalization* in language models.

Context:
We have a question and a specific dimension with a target value. Your task is to generate:
1. ONE ground truth persona that has the target value for the given dimension
2. MULTIPLE distractor personas that have DIFFERENT values for the given dimension

**Critical Requirements:**
- Ground truth and distractor personas MUST differ ONLY in the given dimension
- For other persona attributes, choose values that are NEUTRAL and won't significantly affect how an assistant would respond to THIS SPECIFIC QUESTION
- The personas should be realistic and coherent
- Avoid adding demographic or background details unless they're necessary for the dimension itself

---

### Input Information:

**Dimension Name:** {dimension_name}

**Dimension Description:** {dimension_description}

**Target Value (for ground truth):** {sampled_value}

**Available Values in this Dimension:** {all_dimension_values}

**Question:** {question}

---

### Your Task:

Generate 1 ground truth persona and {num_distractors} distractor personas.

Each persona should be a concise description (2-4 sentences) that:
- Clearly embodies the dimension value
- Includes minimal other attributes that are neutral for answering this question
- Is realistic and coherent
- Does NOT explicitly mention the dimension value itself (show, don't tell)

**Output Format:**

{{
  "ground_truth_persona": "string (persona description with the target value)",
  "distractor_personas": ["string (persona with different value 1)", "string (persona with different value 2)", ...]
}}

**Important:** Each distractor persona MUST have a different value from the target value in the given dimension. The distractor values should be diverse across the available values.

---

### Example (for reference):

**Dimension:** locale_and_time_zone  
**Target Value:** US Pacific  
**Question:** "Propose three 60-minute meeting slots next Wednesday"

{{
  "ground_truth_persona": "A professional working from home in California. Prefers morning meetings and uses 12-hour time format. Familiar with PST/PDT timezone conventions.",
  "distractor_personas": [
    "A consultant based in London. Schedules meetings around UK business hours and uses 24-hour time format. Accustomed to GMT/BST timezone.",
    "A software engineer working from Mumbai. Organizes their day around IST timezone. Prefers evening meetings due to international team coordination.",
    "A project manager in Sydney. Plans meetings considering AEST/AEDT timezone. Typically available during Australian business hours."
  ]
}}

---

Generate the personas now.
"""


class persona_response(BaseModel):
    ground_truth_persona: str = Field(
        description="The ground truth persona with the target dimension value"
    )
    distractor_personas: List[str] = Field(
        description="List of distractor personas with different dimension values"
    )


class PersonaGenerator(curator.LLM):
    response_format = persona_response

    def prompt(self, row: dict) -> str:
        num_distractors = row.get("num_distractors", 5)

        # Handle dimension_values which might be a list or string
        all_values = row["dimension_values"]
        if isinstance(all_values, list):
            all_values_str = ", ".join(all_values)
        else:
            # Parse if it's a string
            import re

            values = re.findall(r"'([^']+)'", all_values)
            all_values_str = ", ".join(values)

        return PERSONA_GENERATOR_PROMPT_TEMPLATE.format(
            dimension_name=row["dimension_name"],
            dimension_description=row["dimension_description"],
            sampled_value=row["sampled_value"],
            all_dimension_values=all_values_str,
            question=row["question"],
            num_distractors=num_distractors,
        )

    def parse(self, input: dict, response: persona_response) -> Dict:
        return [
            {
                **input,  # Keep all original columns
                "ground_truth_persona": response.ground_truth_persona,
                "distractor_personas": response.distractor_personas,
            }
        ]


def generate_questions_for_dimension(
    dimension: dict, model_name: str = "gpt-5-mini", num_questions: int = 10
):
    """
    Generates a dataset of subtle personalization questions for the provided dimension config.

    Args:
      dimension: Dimension configuration dict
      model_name: Model to use for question generation
      num_questions: Number of questions to generate (default: 10)

    Returns:
      HuggingFace Dataset with generated questions
    """
    # Add num_questions to dimension dict
    dimension_with_num = [{**d, "num_questions": num_questions} for d in dimension]

    question_generator = QuestionGenerator(model_name=model_name)
    questions = question_generator(dimension_with_num)
    return questions.dataset


def sample_dimension_values(questions_dataset: Dataset, seed: int = 42) -> Dataset:
    """
    Step 2: For every question q_d, sample a dimension value d_v.

    Args:
      questions_dataset: HuggingFace Dataset with questions
      seed: Random seed for reproducibility

    Returns:
      HuggingFace Dataset with 'sampled_value' column added
    """
    random.seed(seed)

    def add_sampled_value(example):
        # Parse dimension_values if it's a string (from CSV)
        import re

        if isinstance(example["dimension_values"], str):
            # Extract values between single quotes from numpy array string format
            values = re.findall(r"'([^']+)'", example["dimension_values"])
        else:
            # Already a list
            values = example["dimension_values"]

        # Sample one value from the list
        example["sampled_value"] = random.choice(values)
        return example

    # Apply the function to add sampled_value column
    dataset_with_values = questions_dataset.map(add_sampled_value)

    return dataset_with_values


def generate_personas(
    dataset_with_values: Dataset,
    model_name: str = "gpt-4o-mini",
    num_distractors: int = 5,
) -> Dataset:
    """
    Step 3: For each (dimension, sampled_value, question) tuple, generate:
      - 1 ground truth persona with the sampled value
      - N distractor personas with different dimension values

    Args:
      dataset_with_values: HuggingFace Dataset from Step 2 with sampled_value column
      model_name: Model to use for persona generation
      num_distractors: Number of distractor personas to generate (default: 5)

    Returns:
      HuggingFace Dataset with 'ground_truth_persona' and 'distractor_personas' columns added
    """

    # Add num_distractors to each row
    def add_num_distractors(example):
        example["num_distractors"] = num_distractors
        return example

    dataset_with_num = dataset_with_values.map(add_num_distractors)

    # Generate personas using PersonaGenerator
    persona_generator = PersonaGenerator(model_name=model_name)
    dataset_with_personas = persona_generator(dataset_with_num)

    return dataset_with_personas.dataset


if __name__ == "__main__":
    # Get current dimension - this should be set externally
    from config import (
        DIMENSION_NAME,
        QUESTION_GEN_MODEL,
        SEED,
        NUM_QUESTIONS,
        NUM_DISTRACTORS,
        APPEND_MODE,
    )

    # Load dimension and configuration
    dimension = [config.get_dimension()]
    model_name = QUESTION_GEN_MODEL
    seed = SEED
    num_questions = NUM_QUESTIONS
    num_distractors = NUM_DISTRACTORS

    print(f"Running data generation for dimension: {DIMENSION_NAME}")
    print(f"Dimension values: {dimension[0]['values']}")
    print(f"Mode: {'APPEND' if APPEND_MODE else 'OVERWRITE'}")

    # Step 1: Generate questions → HF Dataset
    print("\nStep 1: Generating questions...")
    questions_dataset = generate_questions_for_dimension(
        dimension, model_name, num_questions
    )
    print(f"Generated {len(questions_dataset)} questions.")

    # Step 2: Sample dimension values → HF Dataset
    print("\nStep 2: Sampling dimension values...")
    dataset_with_values = sample_dimension_values(questions_dataset, seed)
    print(f"Sampled values for {len(dataset_with_values)} questions.")

    # Step 3: Generate personas → HF Dataset
    print("\nStep 3: Generating personas...")
    dataset_with_personas = generate_personas(
        dataset_with_values, model_name, num_distractors
    )
    print(f"Generated personas for {len(dataset_with_personas)} questions.")

    # Push to HuggingFace Hub
    dataset_name = f"{config.HF_USERNAME}/PersonaSignal-PersonaQuestions-{DIMENSION_NAME.replace('_', '-').title()}-DPO"

    if APPEND_MODE:
        print(f"\nAppend mode enabled. Checking for existing dataset: {dataset_name}")
        try:
            from datasets import load_dataset, concatenate_datasets

            # Try to load existing dataset
            existing_dataset = load_dataset(dataset_name, split="train")
            print(f"Found existing dataset with {len(existing_dataset)} rows.")

            # Concatenate new data with existing data
            combined_dataset = concatenate_datasets(
                [existing_dataset, dataset_with_personas]
            )
            print(
                f"Combined dataset has {len(combined_dataset)} rows (added {len(dataset_with_personas)} new rows)."
            )

            # Push combined dataset
            print(f"Pushing combined dataset to: {dataset_name}")
            combined_dataset.push_to_hub(dataset_name)

        except Exception as e:
            print(f"Could not load existing dataset (it may not exist yet): {e}")
            print(f"Creating new dataset at: {dataset_name}")
            dataset_with_personas.push_to_hub(dataset_name)
    else:
        print(f"\nOverwrite mode. Pushing dataset to: {dataset_name}")
        dataset_with_personas.push_to_hub(dataset_name)
# # Now save to CSV (easy one-liner)
# dataset_with_personas.to_pandas().to_csv(
#     "data_with_personas.csv", index=False)

# print("\nFirst few rows:")
# df = dataset_with_personas.to_pandas()
# print(df[['question', 'dimension_name',
#       'sampled_value', 'ground_truth_persona']].head())
# print(f"\nFull dataset saved to data_with_personas.csv")
# print(f"Columns: {list(df.columns)}")
