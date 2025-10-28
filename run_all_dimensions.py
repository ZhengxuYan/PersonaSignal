"""
Run the complete pipeline (datagen, collect_response, test_perceivability)
for all dimensions in config.DIMENSIONS.
"""

import subprocess
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
import config


def run_stage(stage_name: str, env=None):
    """Run a single stage of the pipeline."""
    print(f"\n{'='*70}")
    print(f"Running {stage_name}")
    print(f"{'='*70}\n")

    result = subprocess.run(
        ["python", f"inference/{stage_name}.py"], capture_output=False, env=env
    )
    return result.returncode == 0


def run_all_dimensions():
    """Run all 3 stages for all dimensions."""
    dimensions = list(config.DIMENSIONS.keys())

    print(f"\n{'='*70}")
    print(f"Running Pipeline for {len(dimensions)} Dimensions")
    print(f"{'='*70}\n")
    print(f"Dimensions to process: {dimensions}\n")

    results = {}

    for dimension in dimensions:
        print(f"\n{'#'*70}")
        print(f"Processing Dimension: {dimension}")
        print(f"{'#'*70}\n")

        # Create environment with dimension override
        env = os.environ.copy()
        env["PERSONA_DIMENSION"] = dimension

        dim_results = {}

        # Stage 1: Generate questions and personas
        result = subprocess.run(
            ["python", "datagen/datagen.py"], capture_output=False, env=env
        )
        success1 = result.returncode == 0
        dim_results["datagen"] = success1
        if not success1:
            print(f"\n✗ Failed at datagen stage for {dimension}")
            results[dimension] = dim_results
            continue

        # Stage 2: Generate personalized responses
        success2 = run_stage("collect_response", env=env)
        dim_results["collect_response"] = success2
        if not success2:
            print(f"\n✗ Failed at collect_response stage for {dimension}")
            results[dimension] = dim_results
            continue

        # Stage 3: Test perceivability
        success3 = run_stage("test_perceivability", env=env)
        dim_results["test_perceivability"] = success3
        if not success3:
            print(f"\n✗ Failed at test_perceivability stage for {dimension}")
            results[dimension] = dim_results
            continue

        print(f"\n✓ Successfully completed all stages for {dimension}")
        results[dimension] = dim_results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    for dimension, dim_results in results.items():
        if all(dim_results.values()):
            print(f"✓ {dimension}: All stages completed")
        else:
            print(
                f"✗ {dimension}: Failed stages:",
                [k for k, v in dim_results.items() if not v],
            )

    # Combine all datasets
    print(f"\n{'='*70}")
    print("Combining Datasets from All Dimensions")
    print(f"{'='*70}\n")

    # Reset to first dimension for combine (doesn't matter which one)
    config.DIMENSION_NAME = dimensions[0]

    try:
        print("Running combine_datasets_selective.py...")
        subprocess.run(["python", "combine_datasets_selective.py"], check=False)
        print("\n✓ Combined datasets successfully")
    except Exception as e:
        print(f"Warning: Combining datasets failed: {e}")

    # Run analysis on combined dataset
    print(f"\n{'='*70}")
    print("Running Analysis on Combined Dataset")
    print(f"{'='*70}\n")

    try:
        print("Running analyze_perceivability.py...")
        subprocess.run(["python", "analyze_perceivability.py"], check=False)
        print("\n✓ Analysis completed")
    except Exception as e:
        print(f"Warning: Analysis failed: {e}")

    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_all_dimensions()
