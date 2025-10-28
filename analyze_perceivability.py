"""
Simple accuracy analysis for PersonaSignal perceivability results.
Shows accuracy for each dimension and overall.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset as load_dataset_hf

# Set publication-quality plot style
plt.rcParams.update(
    {
        "font.size": 14,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2,
    }
)

sys.path.insert(0, str(Path(__file__).parent))
import config


def load_perceivability_data(
    dataset_name: str = None, response_model: str = None, judge_model: str = None
):
    """
    Load perceivability dataset.

    Args:
        dataset_name: Full dataset name. If None, constructs from config.
        response_model: Response model to analyze. If None, uses config default.
        judge_model: Judge model that was used. If None, uses config default.
    """
    if dataset_name is None:
        # Use the default combined dataset name with response model suffix
        response_model = response_model or config.RESPONSE_GEN_MODEL
        judge_model = judge_model or config.JUDGE_MODEL
        dataset_name = (
            f"{config.HF_USERNAME}/PersonaSignal-All-Perceivability-{response_model}"
        )

    print(f"Loading: {dataset_name}")
    dataset = load_dataset_hf(dataset_name, split="train")
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} rows")

    # Show model information if available
    if "judge_model" in df.columns:
        print("\nModel Configuration:")
        for col in [
            "question_gen_model",
            "persona_gen_model",
            "response_gen_model",
            "judge_model",
        ]:
            if col in df.columns:
                models = df[col].unique()
                print(f"  {col.replace('_', ' ').title()}: {', '.join(models)}")

    return df


def plot_accuracy(df: pd.DataFrame, output_filename: str = "accuracy.png"):
    # Calculate accuracy by dimension
    dim_acc = df.groupby("dimension_name")["reward"].mean() * 100

    # Calculate overall accuracy
    overall_acc = df["reward"].mean() * 100

    # Prepare data for plotting with formatted labels
    dim_labels = [dim.replace("_", " ").title() for dim in dim_acc.index]
    categories = ["Overall"] + dim_labels
    accuracies = [overall_acc] + list(dim_acc.values)

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define color palette (professional academic colors)
    color_overall = "#2C3E50"  # Dark blue-gray for overall
    color_dims = "#3498DB"  # Professional blue for dimensions

    colors = [color_overall] + [color_dims] * len(dim_labels)

    # Create bars with edge colors for definition
    bars = ax.bar(
        range(len(categories)),
        accuracies,
        width=0.6,  # Make bars narrower
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        alpha=0.85,
        zorder=3,
    )

    # Add value labels on bars with better positioning
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.5,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    # Styling
    ax.set_ylabel("Accuracy (%)", fontsize=16, fontweight="bold", labelpad=10)
    ax.set_xlabel("Dimension", fontsize=16, fontweight="bold", labelpad=10)

    # Build title with model information
    title = "Perceivability Test: Judge Accuracy by Dimension"
    if "response_gen_model" in df.columns and "judge_model" in df.columns:
        response_models = df["response_gen_model"].unique()
        judge_models = df["judge_model"].unique()
        if len(response_models) == 1 and len(judge_models) == 1:
            subtitle = (
                f"Response Model: {response_models[0]} | Judge Model: {judge_models[0]}"
            )
            ax.text(
                0.5,
                1.05,
                subtitle,
                transform=ax.transAxes,
                ha="center",
                fontsize=13,
                style="italic",
                color="#555555",
            )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=30)

    # Set y-axis limits with some padding
    ax.set_ylim(0, 105)

    # Improve grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="gray", zorder=0)
    ax.set_axisbelow(True)

    # Set x-axis
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha="right")

    # Add horizontal line at 100% for reference
    ax.axhline(y=100, color="gray", linestyle="-", linewidth=1, alpha=0.3, zorder=1)

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Increase spine width for remaining spines
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n✓ Saved: {output_filename}")
    plt.close()

    return overall_acc, dim_acc


def print_stats(overall_acc, dim_acc, df=None):
    print(f"\nOverall Accuracy: {overall_acc:.1f}%")
    print("\nAccuracy by Dimension:")
    for dim, acc in dim_acc.items():
        print(f'  {dim.replace("_", " ").title()}: {acc:.1f}%')

    # Show accuracy breakdown by model if available
    if df is not None and "judge_model" in df.columns:
        judge_models = df["judge_model"].unique()
        if len(judge_models) > 1:
            print("\nAccuracy by Judge Model:")
            judge_acc = df.groupby("judge_model")["reward"].mean() * 100
            for model, acc in judge_acc.items():
                count = len(df[df["judge_model"] == model])
                print(f"  {model}: {acc:.1f}% ({count} rows)")

        response_models = (
            df["response_gen_model"].unique()
            if "response_gen_model" in df.columns
            else []
        )
        if len(response_models) > 1:
            print("\nAccuracy by Response Generation Model:")
            resp_acc = df.groupby("response_gen_model")["reward"].mean() * 100
            for model, acc in resp_acc.items():
                count = len(df[df["response_gen_model"] == model])
                print(f"  {model}: {acc:.1f}% ({count} rows)")


def compare_response_models(response_models: list[str], judge_model: str = None):
    """
    Compare multiple response models side by side.

    Args:
        response_models: List of response model names to compare
        judge_model: Judge model to use (default: config.JUDGE_MODEL)
    """
    if judge_model is None:
        judge_model = config.JUDGE_MODEL

    print(f"\n{'='*60}")
    print(f"Comparing Response Models (Judge: {judge_model})")
    print(f"Models: {response_models}")
    print(f"{'='*60}")

    results = {}

    for response_model in response_models:
        print(f"\n--- Loading dataset for {response_model} ---")
        dataset_name = (
            f"{config.HF_USERNAME}/PersonaSignal-All-Perceivability-{response_model}"
        )

        try:
            df = load_perceivability_data(
                dataset_name=dataset_name,
                response_model=response_model,
                judge_model=judge_model,
            )

            model_safe = response_model.replace("-", "_").replace(".", "_")
            output_file = f"accuracy_{model_safe}.png"
            overall, dims = plot_accuracy(df, output_filename=output_file)
            print_stats(overall, dims, df)

            results[response_model] = {"overall": overall, "dimensions": dims}
        except Exception as e:
            print(f"✗ Failed to load {response_model}: {e}")

    # Print comparison summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print("\nOverall Accuracy:")
        for model, data in results.items():
            print(f"  {model}: {data['overall']:.1f}%")

    return results


def main():
    """
    Main analysis function.

    To analyze a single response model (default from config):
        python analyze_perceivability.py

    To compare multiple response models, uncomment and configure below.
    """
    # ========== CONFIGURE HERE ==========

    # Option 1: Analyze single response model (uses config defaults)
    response_model = config.RESPONSE_GEN_MODEL
    judge_model = config.JUDGE_MODEL

    # Option 2: Compare multiple response models
    # Uncomment the following to compare different response models:
    # compare_models = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"]
    # compare_response_models(compare_models, judge_model=judge_model)
    # return

    # =====================================

    # Load and analyze single model
    dataset_name = (
        f"{config.HF_USERNAME}/PersonaSignal-All-Perceivability-{response_model}"
    )
    df = load_perceivability_data(
        dataset_name=dataset_name,
        response_model=response_model,
        judge_model=judge_model,
    )

    overall, dims = plot_accuracy(df)
    print_stats(overall, dims, df)


if __name__ == "__main__":
    main()
