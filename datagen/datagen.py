import random
from typing import Dict, List

from datasets import Dataset
from utils import PersonaGenerator, QuestionGenerator


def generate_questions_for_dimension(dimension: dict, model_name: str = "gpt-5-mini"):
  """
  Generates a dataset of subtle personalization questions for the provided dimension config.
  Returns the dataset (e.g., a Bespoke Dataset object).
  """
  question_generator = QuestionGenerator(model_name=model_name)
  questions = question_generator(dimension)
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
    if isinstance(example['dimension_values'], str):
      # Extract values between single quotes from numpy array string format
      values = re.findall(r"'([^']+)'", example['dimension_values'])
    else:
      # Already a list
      values = example['dimension_values']

    # Sample one value from the list
    example['sampled_value'] = random.choice(values)
    return example

  # Apply the function to add sampled_value column
  dataset_with_values = questions_dataset.map(add_sampled_value)

  return dataset_with_values


def generate_personas(dataset_with_values: Dataset, model_name: str = "gpt-4o-mini", num_distractors: int = 5) -> Dataset:
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
    example['num_distractors'] = num_distractors
    return example

  dataset_with_num = dataset_with_values.map(add_num_distractors)

  # Generate personas using PersonaGenerator
  persona_generator = PersonaGenerator(model_name=model_name)
  dataset_with_personas = persona_generator(dataset_with_num)

  return dataset_with_personas.dataset


if __name__ == "__main__":
  # Define dimension (changed "notes" to "description")
  dimension = [{
      "name": "locale_and_time_zone",
      "values": [
          "US Pacific", "US Eastern", "UK", "EU Central", "India",
          "China Mainland", "Japan", "Brazil", "Australia", "Africa"
      ],
      "description": "Controls the user's geographic location, timezone, date/time formatting preferences, and cultural conventions for calendar, currency, and measurements."
  }]

  # Configuration
  model_name = "gpt-5"
  seed = 42
  num_distractors = 5

  # Step 1: Generate questions → HF Dataset
  print("Step 1: Generating questions...")
  questions_dataset = generate_questions_for_dimension(dimension, model_name)
  print(f"Generated {len(questions_dataset)} questions.")

  # Step 2: Sample dimension values → HF Dataset
  print("\nStep 2: Sampling dimension values...")
  dataset_with_values = sample_dimension_values(questions_dataset, seed)
  print(f"Sampled values for {len(dataset_with_values)} questions.")

  # Step 3: Generate personas → HF Dataset
  print("\nStep 3: Generating personas...")
  dataset_with_personas = generate_personas(
      dataset_with_values, model_name, num_distractors)
  print(f"Generated personas for {len(dataset_with_personas)} questions.")

  # Now save to CSV (easy one-liner)
  dataset_with_personas.to_pandas().to_csv(
      "data_with_personas.csv", index=False)

  print("\nFirst few rows:")
  df = dataset_with_personas.to_pandas()
  print(df[['question', 'dimension_name',
        'sampled_value', 'ground_truth_persona']].head())
  print(f"\nFull dataset saved to data_with_personas.csv")
  print(f"Columns: {list(df.columns)}")
