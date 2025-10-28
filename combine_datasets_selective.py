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
                "question_gen_model": Value("string"),
                "persona_gen_model": Value("string"),
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
                "question_gen_model": Value("string"),
                "persona_gen_model": Value("string"),
                "response_gen_model": Value("string"),
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
                "question_gen_model": Value("string"),
                "persona_gen_model": Value("string"),
                "response_gen_model": Value("string"),
                "judge_model": Value("string"),
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
                # Get dataset name with model suffix
                if stage == "questions":
                    dataset_name = config.get_dataset_name_with_model(
                        stage, config.QUESTION_GEN_MODEL
                    )
                elif stage == "responses":
                    dataset_name = config.get_dataset_name_with_model(
                        stage, config.RESPONSE_GEN_MODEL
                    )
                else:  # perceivability
                    dataset_name = config.get_dataset_name_with_model(
                        stage, config.JUDGE_MODEL
                    )

                print(f"  Loading: {dataset_name}")

                dataset = load_dataset(dataset_name, split="train")

                # Add model tracking fields
                if stage == "questions":
                    dataset = dataset.add_column(
                        "question_gen_model", [config.QUESTION_GEN_MODEL] * len(dataset)
                    )
                    dataset = dataset.add_column(
                        "persona_gen_model", [config.PERSONA_GEN_MODEL] * len(dataset)
                    )
                elif stage == "responses":
                    dataset = dataset.add_column(
                        "question_gen_model", [config.QUESTION_GEN_MODEL] * len(dataset)
                    )
                    dataset = dataset.add_column(
                        "persona_gen_model", [config.PERSONA_GEN_MODEL] * len(dataset)
                    )
                    dataset = dataset.add_column(
                        "response_gen_model", [config.RESPONSE_GEN_MODEL] * len(dataset)
                    )
                else:  # perceivability
                    dataset = dataset.add_column(
                        "question_gen_model", [config.QUESTION_GEN_MODEL] * len(dataset)
                    )
                    dataset = dataset.add_column(
                        "persona_gen_model", [config.PERSONA_GEN_MODEL] * len(dataset)
                    )
                    dataset = dataset.add_column(
                        "response_gen_model", [config.RESPONSE_GEN_MODEL] * len(dataset)
                    )
                    dataset = dataset.add_column(
                        "judge_model", [config.JUDGE_MODEL] * len(dataset)
                    )

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
        # Add model suffix to output name
        # For perceivability, use response model since we compare different response models
        if stage == "questions":
            model_suffix = config.QUESTION_GEN_MODEL
        elif stage == "responses":
            model_suffix = config.RESPONSE_GEN_MODEL
        else:  # perceivability
            model_suffix = config.RESPONSE_GEN_MODEL

        output_name = f"{config.HF_USERNAME}/PersonaSignal-{suffix}-{stage.title()}-{model_suffix}"
        print(f"Pushing: {output_name}")
        print(f"  Rows: {len(dataset)}")
        print(f"  Dimensions: {num_dims}")

        # Show model tracking info
        if stage == "questions":
            print(
                f"  Models tracked: question={config.QUESTION_GEN_MODEL}, persona={config.PERSONA_GEN_MODEL}"
            )
        elif stage == "responses":
            print(
                f"  Models tracked: question={config.QUESTION_GEN_MODEL}, persona={config.PERSONA_GEN_MODEL}, response={config.RESPONSE_GEN_MODEL}"
            )
        else:  # perceivability
            print(
                f"  Models tracked: question={config.QUESTION_GEN_MODEL}, persona={config.PERSONA_GEN_MODEL}, response={config.RESPONSE_GEN_MODEL}, judge={config.JUDGE_MODEL}"
            )

        print(f"  Columns: {dataset.column_names}")

        dataset.push_to_hub(output_name)
        print("  ✓ Success!\n")

    print("Done!")


if __name__ == "__main__":
    main()
