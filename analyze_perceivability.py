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
    dataset_name: str = None,
    response_model: str = None,
    judge_model: str = None,
    hf_username: str = None,
):
    """
    Load perceivability dataset.

    Args:
        dataset_name: Full dataset name. If None, constructs from config.
        response_model: Response model to analyze. If None, uses config default.
        judge_model: Judge model that was used. If None, uses config default.
        hf_username: HuggingFace username/account. If None, uses config default.
    """
    if dataset_name is None:
        # Use the default combined dataset name with response model suffix
        response_model = response_model or config.RESPONSE_GEN_MODEL
        judge_model = judge_model or config.JUDGE_MODEL
        hf_username = hf_username or config.HF_USERNAME
        dataset_name = (
            f"{hf_username}/PersonaSignal-All-Perceivability-{response_model}"
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


def plot_comparison(
    results: dict, judge_model: str, output_filename: str = "accuracy_comparison.png"
):
    """
    Plot comparison of multiple models in a single graph.

    Args:
        results: Dict mapping model_name -> {"overall": float, "dimensions": pd.Series}
        judge_model: Name of the judge model used
        output_filename: Output file name
    """
    import numpy as np

    # Get dimensions that are common across all models
    all_dimensions = [
        set(results[model]["dimensions"].index) for model in results.keys()
    ]
    dimensions = sorted(list(set.intersection(*all_dimensions)))
    dim_labels = [dim.replace("_", " ").title() for dim in dimensions]

    # Prepare data - sort models by overall accuracy (lowest to highest)
    models = sorted(results.keys(), key=lambda m: results[m]["overall"])
    categories = ["Overall"] + dim_labels
    n_categories = len(categories)
    n_models = len(models)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors for different models
    color_palette = ["#2C3E50", "#E74C3C", "#27AE60", "#F39C12", "#9B59B6", "#1ABC9C"]
    colors = color_palette[:n_models]

    # Set bar width and positions
    bar_width = 0.8 / n_models
    x = np.arange(n_categories)

    # Plot bars for each model
    for i, (model, color) in enumerate(zip(models, colors)):
        offset = (i - n_models / 2 + 0.5) * bar_width

        # Get accuracies for this model
        overall = results[model]["overall"]
        dim_accs = [results[model]["dimensions"][dim] for dim in dimensions]
        accuracies = [overall] + dim_accs

        bars = ax.bar(
            x + offset,
            accuracies,
            bar_width,
            label=model,
            color=color,
            edgecolor="black",
            linewidth=1,
            alpha=0.85,
            zorder=3,
        )

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            # Only show label if bar is tall enough and not too crowded
            if height > 5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{acc:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    # Styling
    ax.set_ylabel("Accuracy (%)", fontsize=16, fontweight="bold", labelpad=10)
    ax.set_xlabel("Dimension", fontsize=16, fontweight="bold", labelpad=10)

    # Title with judge model info
    title = "Perceivability Test: Model Comparison"
    subtitle = f"Judge Model: {judge_model}"
    ax.set_title(title, fontsize=18, fontweight="bold", pad=40)
    ax.text(
        0.5,
        1.08,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        fontsize=13,
        style="italic",
        color="#555555",
    )

    # Set y-axis limits
    ax.set_ylim(0, 110)

    # Grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="gray", zorder=0)
    ax.set_axisbelow(True)

    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")

    # Add horizontal line at 100%
    ax.axhline(y=100, color="gray", linestyle="-", linewidth=1, alpha=0.3, zorder=1)

    # Legend - position it below the plot to avoid overlapping with bars and x-labels
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),  # Move further down to clear rotated labels
        ncol=min(n_models, 3),  # Use up to 3 columns for horizontal layout
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=12,
        title="Response Model",
        title_fontsize=13,
    )

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # Use tight_layout with extra padding, then save with bbox_inches='tight' to include legend
    plt.tight_layout()
    plt.savefig(
        output_filename, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.3
    )
    print(f"\n✓ Saved comparison: {output_filename}")
    plt.close()


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


def compare_response_models(
    response_models: list[str],
    judge_model: str = None,
    model_hf_accounts: dict[str, str] = None,
    generate_individual_plots: bool = True,
):
    """
    Compare multiple response models side by side.

    Args:
        response_models: List of response model names to compare
        judge_model: Judge model to use (default: config.JUDGE_MODEL)
        model_hf_accounts: Dict mapping response model names to HF usernames.
                          Example: {"gpt-4o": "JasonYan777", "claude-3": "other_username"}
                          If None or model not in dict, uses config.HF_USERNAME
        generate_individual_plots: If True, also generate individual plots for each model
    """
    if judge_model is None:
        judge_model = config.JUDGE_MODEL

    if model_hf_accounts is None:
        model_hf_accounts = {}

    print(f"\n{'='*60}")
    print(f"Comparing Response Models (Judge: {judge_model})")
    print(f"Models: {response_models}")
    print(f"{'='*60}")

    results = {}

    for response_model in response_models:
        print(f"\n--- Loading dataset for {response_model} ---")

        # Get HF username for this model
        hf_username = model_hf_accounts.get(response_model, config.HF_USERNAME)
        dataset_name = (
            f"{hf_username}/PersonaSignal-All-Perceivability-{response_model}"
        )

        try:
            df = load_perceivability_data(
                dataset_name=dataset_name,
                response_model=response_model,
                judge_model=judge_model,
                hf_username=hf_username,
            )

            # Generate individual plot if requested
            if generate_individual_plots:
                model_safe = response_model.replace("-", "_").replace(".", "_")
                output_file = f"accuracy_{model_safe}.png"
                overall, dims = plot_accuracy(df, output_filename=output_file)
                print_stats(overall, dims, df)
            else:
                # Just calculate stats without plotting
                overall = df["reward"].mean() * 100
                dims = df.groupby("dimension_name")["reward"].mean() * 100
                print_stats(overall, dims, df)

            results[response_model] = {"overall": overall, "dimensions": dims}
        except Exception as e:
            print(f"✗ Failed to load {response_model}: {e}")

    # Generate combined comparison plot
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("GENERATING COMBINED COMPARISON")
        print(f"{'='*60}")

        plot_comparison(results, judge_model, output_filename="accuracy_comparison.png")

        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print("\nOverall Accuracy:")
        for model, data in results.items():
            print(f"  {model}: {data['overall']:.1f}%")

        # Show dimension-by-dimension comparison
        print("\nAccuracy by Dimension:")
        all_dims = list(results[list(results.keys())[0]]["dimensions"].index)
        for dim in all_dims:
            dim_label = dim.replace("_", " ").title()
            print(f"\n  {dim_label}:")
            for model, data in results.items():
                acc = data["dimensions"][dim]
                print(f"    {model}: {acc:.1f}%")

    return results


def analyze_single_dimension(
    dimension_name: str = None,
    response_model: str = None,
    hf_username: str = None,
    output_filename: str = None,
):
    """
    Analyze a single dimension dataset.

    Args:
        dimension_name: Dimension to analyze (e.g., "planning_horizon").
                       If None, uses config.DIMENSION_NAME
        judge_model: Judge model name (e.g., "gpt-4o"). If None, uses config.JUDGE_MODEL
        hf_username: HuggingFace username. If None, uses config.HF_USERNAME
        output_filename: Custom output filename. If None, auto-generates based on dimension and model
    """
    if dimension_name is None:
        dimension_name = config.DIMENSION_NAME
    if response_model is None:
        response_model = config.RESPONSE_GEN_MODEL
    if hf_username is None:
        hf_username = config.HF_USERNAME

    # Format dimension name for dataset (e.g., "planning_horizon" -> "Planning-Horizon")
    dimension_formatted = dimension_name.replace("_", "-").title()

    # Construct dataset name
    dataset_name = f"{hf_username}/PersonaSignal-PerceivabilityTest-{dimension_formatted}-{response_model}"

    print(f"\n{'='*60}")
    print(f"Analyzing Single Dimension: {dimension_name}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    # Load data
    df = load_perceivability_data(
        dataset_name=dataset_name,
        response_model=response_model,
        hf_username=hf_username,
    )

    # Generate output filename if not provided
    if output_filename is None:
        dimension_safe = dimension_name.replace("_", "-")
        response_safe = response_model.replace("/", "-").replace(".", "_")
        output_filename = f"accuracy_{dimension_safe}_{response_safe}.png"

    # Plot accuracy (for single dimension, there's only one dimension category)
    overall, dims = plot_accuracy(df, output_filename=output_filename)
    print_stats(overall, dims, df)

    return df, overall, dims


def main():
    """
    Main analysis function.

    To analyze a single response model (default from config):
        python analyze_perceivability.py

    To compare multiple response models, uncomment and configure below.
    """
    # ========== CONFIGURE HERE ==========

    # Option 1: Analyze single response model with ALL dimensions combined (uses config defaults)
    response_model = config.RESPONSE_GEN_MODEL.split("/")[-1]
    judge_model = config.JUDGE_MODEL
    hf_username = config.HF_USERNAME  # Or specify different account

    # Option 2: Compare multiple response models from different HF accounts
    # Uncomment the following to compare different response models:
    # This will generate:
    # - Individual plots for each model (accuracy_gpt_4o_mini.png, accuracy_gpt_4o.png)
    # - A combined comparison plot (accuracy_comparison.png)
    compare_models = [
        "gpt-4o-mini",
        "gpt-4o",
        "Meta-Llama-3.1-8B-Instruct-Turbo",
        "claude-sonnet-4-5-20250929",
    ]
    model_accounts = {
        "gpt-4o-mini": "JasonYan777",
        "gpt-4o": "JasonYan777",
        "Meta-Llama-3.1-8B-Instruct-Turbo": "JasonYan777",
        "claude-sonnet-4-5-20250929": "JasonYan777",
        # "gpt-5": "JasonYan777",
    }
    compare_response_models(
        compare_models,
        judge_model=judge_model,
        model_hf_accounts=model_accounts,
        generate_individual_plots=True,  # Set to False to only generate comparison plot
    )
    return

    # Option 3: Analyze a single dimension dataset
    # Uncomment the following to analyze a specific dimension:
    # analyze_single_dimension(
    #     dimension_name="programming_expertise",  # or any dimension from config.DIMENSIONS
    #     response_model=config.RESPONSE_GEN_MODEL,
    #     hf_username=config.HF_USERNAME,
    #     output_filename=f"accuracy_programming_expertise_{config.RESPONSE_GEN_MODEL.split('/')[-1]}.png",  # optional
    # )
    # return


if __name__ == "__main__":
    main()
