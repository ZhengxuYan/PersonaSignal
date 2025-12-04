"""
Run the complete pipeline (datagen, collect_response, test_perceivability)
for all dimensions in config.DIMENSIONS.
"""

import argparse
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
    
    script_name = f"inference/{stage_name}.py"
    
    # Check if we should use the Tinker inference script
    if stage_name == "collect_response" and config.RESPONSE_GEN_MODEL.startswith("DPO-Tinker"):
        print("Detected Tinker model, using inference/collect_response_tinker.py")
        script_name = "inference/collect_response_tinker.py"
        
        # Ensure PYTHONPATH includes tinker-cookbook
        if env is None:
            env = os.environ.copy()
        
        tinker_path = str(Path(__file__).parent / ".venv" / "src" / "tinker-cookbook")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] += f":{tinker_path}"
        else:
            env["PYTHONPATH"] = tinker_path

    result = subprocess.run(
        ["python", script_name], capture_output=False, env=env
    )
    return result.returncode == 0


def run_pipeline(dimensions=None, stages=None, skip_combine=False, skip_analysis=False):
    """
    Run pipeline stages for specified dimensions.

    Args:
        dimensions: List of dimension names to process. If None, processes all dimensions.
        stages: List of stage names to run. Options: 'datagen', 'collect_response', 'test_perceivability'.
                If None, runs all stages.
        skip_combine: If True, skip combining datasets step.
        skip_analysis: If True, skip analysis step.
    """
    # Default to all dimensions
    if dimensions is None:
        dimensions = list(config.DIMENSIONS.keys())

    # Default to all stages
    if stages is None:
        stages = ["datagen", "collect_response", "test_perceivability"]

    # Validate dimensions
    invalid_dims = [d for d in dimensions if d not in config.DIMENSIONS]
    if invalid_dims:
        print(f"Error: Invalid dimensions: {invalid_dims}")
        print(f"Available dimensions: {list(config.DIMENSIONS.keys())}")
        return

    # Validate stages
    valid_stages = ["datagen", "collect_response", "test_perceivability"]
    invalid_stages = [s for s in stages if s not in valid_stages]
    if invalid_stages:
        print(f"Error: Invalid stages: {invalid_stages}")
        print(f"Valid stages: {valid_stages}")
        return

    print(f"\n{'='*70}")
    print(f"Running Pipeline for {len(dimensions)} Dimension(s)")
    print(f"{'='*70}\n")
    print(f"Dimensions to process: {dimensions}")
    print(f"Stages to run: {stages}\n")

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
        if "datagen" in stages:
            result = subprocess.run(
                ["python", "datagen/datagen.py"], capture_output=False, env=env
            )
            success1 = result.returncode == 0
            dim_results["datagen"] = success1
            if not success1:
                print(f"\n✗ Failed at datagen stage for {dimension}")
                results[dimension] = dim_results
                continue
        else:
            dim_results["datagen"] = "skipped"

        # Stage 2: Generate personalized responses
        if "collect_response" in stages:
            success2 = run_stage("collect_response", env=env)
            dim_results["collect_response"] = success2
            if not success2:
                print(f"\n✗ Failed at collect_response stage for {dimension}")
                results[dimension] = dim_results
                continue
        else:
            dim_results["collect_response"] = "skipped"

        # Stage 3: Test perceivability
        if "test_perceivability" in stages:
            success3 = run_stage("test_perceivability", env=env)
            dim_results["test_perceivability"] = success3
            if not success3:
                print(f"\n✗ Failed at test_perceivability stage for {dimension}")
                results[dimension] = dim_results
                continue
        else:
            dim_results["test_perceivability"] = "skipped"

        print(f"\n✓ Successfully completed requested stages for {dimension}")
        results[dimension] = dim_results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    for dimension, dim_results in results.items():
        completed = [k for k, v in dim_results.items() if v is True]
        failed = [k for k, v in dim_results.items() if v is False]
        skipped = [k for k, v in dim_results.items() if v == "skipped"]

        if failed:
            print(f"✗ {dimension}: Failed stages: {failed}")
        elif completed:
            print(f"✓ {dimension}: Completed stages: {completed}")
            if skipped:
                print(f"  Skipped stages: {skipped}")
        else:
            print(f"○ {dimension}: All stages skipped")

    # Combine all datasets
    if not skip_combine:
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
    else:
        print(f"\n{'='*70}")
        print("Skipping dataset combination")
        print(f"{'='*70}\n")

    # Run analysis on combined dataset
    if not skip_analysis:
        print(f"\n{'='*70}")
        print("Running Analysis on Combined Dataset")
        print(f"{'='*70}\n")

        try:
            print("Running analyze_perceivability.py...")
            subprocess.run(["python", "analyze_perceivability.py"], check=False)
            print("\n✓ Analysis completed")
        except Exception as e:
            print(f"Warning: Analysis failed: {e}")
    else:
        print(f"\n{'='*70}")
        print("Skipping analysis")
        print(f"{'='*70}\n")

    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PersonaSignal pipeline for specified dimensions and stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages for all dimensions (default)
  python run_all_dimensions.py

  # Run only last 2 stages for planning_horizon
  python run_all_dimensions.py -d planning_horizon -s collect_response test_perceivability

  # Run only datagen for multiple dimensions
  python run_all_dimensions.py -d programming_expertise locale_and_time_zone -s datagen

  # Run all stages but skip combine and analysis
  python run_all_dimensions.py --skip-combine --skip-analysis
        """,
    )

    parser.add_argument(
        "-d",
        "--dimensions",
        nargs="+",
        choices=list(config.DIMENSIONS.keys()),
        help="Specific dimensions to process. If not specified, processes all dimensions.",
        metavar="DIM",
    )

    parser.add_argument(
        "-s",
        "--stages",
        nargs="+",
        choices=["datagen", "collect_response", "test_perceivability"],
        help="Specific stages to run. If not specified, runs all stages.",
        metavar="STAGE",
    )

    parser.add_argument(
        "--skip-combine", action="store_true", help="Skip the dataset combination step"
    )

    parser.add_argument(
        "--skip-analysis", action="store_true", help="Skip the analysis step"
    )

    args = parser.parse_args()

    run_pipeline(
        dimensions=args.dimensions,
        stages=args.stages,
        skip_combine=args.skip_combine,
        skip_analysis=args.skip_analysis,
    )
