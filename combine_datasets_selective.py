"""
Selectively combine datasets from specific dimensions.
Use this when you want to combine only a subset of dimensions.
"""

import sys
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Features, Value, Sequence
from dotenv import load_dotenv

load_dotenv()

# Import config
sys.path.insert(0, str(Path(__file__).parent))
import config


def combine_specific_dimensions(dimensions: list[str], stages: list[str] = None):
    """
    Combine datasets from specific dimensions.

    Args:
        dimensions: List of dimension names to combine
        stages: List of stages to combine. If None, combines all stages.

    Example:
        combine_specific_dimensions(
            dimensions=["programming_expertise", "agency_expectation"],
            stages=["responses", "perceivability"]
        )
    """
    if stages is None:
        stages = ["questions", "responses", "perceivability"]

    print(f"Combining dimensions: {dimensions}")
    print(f"Stages: {stages}\n")

    # Validate dimensions
    invalid = [d for d in dimensions if d not in config.DIMENSIONS]
    if invalid:
        raise ValueError(
            f"Invalid dimensions: {invalid}. Must be one of {list(config.DIMENSIONS.keys())}"
        )

    # Define common schemas for each stage to handle variable-length lists
    common_schemas = {
        "questions": Features(
            {
                "dimension_name": Value("string"),
                "dimension_values": Sequence(Value("string")),
                "dimension_description": Value("string"),
                "question": Value("string"),
                "why_differ": Value("string"),
                "how_subtle": Value("string"),
                "sampled_value": Value("string"),
                "num_distractors": Value("int64"),
                "ground_truth_persona": Value("string"),
                "distractor_personas": Sequence(Value("string")),
            }
        ),
        "responses": Features(
            {
                "dimension_name": Value("string"),
                "dimension_values": Sequence(Value("string")),
                "dimension_description": Value("string"),
                "question": Value("string"),
                "why_differ": Value("string"),
                "how_subtle": Value("string"),
                "sampled_value": Value("string"),
                "num_distractors": Value("int64"),
                "ground_truth_persona": Value("string"),
                "distractor_personas": Sequence(Value("string")),
                "personalized_response": Value("string"),
            }
        ),
        "perceivability": Features(
            {
                "dimension_name": Value("string"),
                "dimension_values": Sequence(Value("string")),
                "dimension_description": Value("string"),
                "question": Value("string"),
                "why_differ": Value("string"),
                "how_subtle": Value("string"),
                "sampled_value": Value("string"),
                "num_distractors": Value("int64"),
                "ground_truth_persona": Value("string"),
                "distractor_personas": Sequence(Value("string")),
                "personalized_response": Value("string"),
                "judge_choice": Value("string"),
                "judge_rationale": Value("string"),
                "reward": Value("int64"),
            }
        ),
    }

    results = {}

    for stage in stages:
        print(f"\n{'='*60}")
        print(f"Processing stage: {stage}")
        print(f"{'='*60}")

        datasets_to_combine = []

        for dimension in dimensions:
            # Temporarily set dimension to get the correct dataset name
            original_dimension = config.DIMENSION_NAME
            config.DIMENSION_NAME = dimension

            try:
                dataset_name = config.get_dataset_name(stage)
                print(f"  Loading: {dataset_name}")

                dataset = load_dataset(dataset_name, split="train")

                # Cast to common schema to ensure compatibility
                dataset = dataset.cast(common_schemas[stage])

                datasets_to_combine.append(dataset)
                print(f"    ✓ Loaded {len(dataset)} rows")

            except Exception as e:
                print(f"    ✗ Failed: {e}")

            finally:
                config.DIMENSION_NAME = original_dimension

        if datasets_to_combine:
            combined = concatenate_datasets(datasets_to_combine)
            results[stage] = combined
            print(f"\n✓ Combined {len(combined)} rows for {stage}")
        else:
            print(f"\n✗ No datasets found for {stage}")

    return results


def main():
    """Example usage: Combine specific dimensions."""

    # ========== CONFIGURE HERE ==========

    # Option 1: Combine specific dimensions
    # dimensions_to_combine = [
    #     "programming_expertise",
    #     "agency_expectation",
    #     "verification_orientation",
    # ]

    # Option 2: Or combine all available dimensions
    dimensions_to_combine = list(config.DIMENSIONS.keys())

    # Choose which stages to combine
    stages_to_combine = ["questions", "responses", "perceivability"]
    # stages_to_combine = ["responses"]  # Or just specific stages

    # =====================================

    print(f"Available dimensions in config: {list(config.DIMENSIONS.keys())}\n")

    combined_datasets = combine_specific_dimensions(
        dimensions=dimensions_to_combine, stages=stages_to_combine
    )

    # Push to HuggingFace
    print(f"\n{'='*60}")
    print("Pushing combined datasets to HuggingFace...")
    print(f"{'='*60}\n")

    # Generate concise suffix
    num_dims = len(dimensions_to_combine)
    all_dims = list(config.DIMENSIONS.keys())

    if num_dims == len(all_dims):
        # All dimensions - use simple name
        suffix = "All"
    elif num_dims == 1:
        # Single dimension - use dimension name
        suffix = dimensions_to_combine[0].replace("_", "-").title()
    else:
        # Multiple but not all - use count
        suffix = f"{num_dims}D"

    for stage, dataset in combined_datasets.items():
        output_name = f"{config.HF_USERNAME}/PersonaSignal-{suffix}-{stage.title()}"
        print(f"Pushing: {output_name}")
        print(f"  Rows: {len(dataset)}")
        print(f"  Dimensions: {num_dims}")
        print(f"  Columns: {dataset.column_names}")

        dataset.push_to_hub(output_name)
        print(f"  ✓ Success!\n")

    print("Done!")


if __name__ == "__main__":
    main()
