"""
Simple accuracy analysis for PersonaSignal perceivability results.
Shows accuracy for each dimension and overall.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset as load_dataset_hf

sys.path.insert(0, str(Path(__file__).parent))
import config


def load_perceivability_data(dataset_name: str = None):
    if dataset_name is None:
        dataset_name = f"{config.HF_USERNAME}/PersonaSignal-All-Perceivability"

    print(f"Loading: {dataset_name}")
    dataset = load_dataset_hf(dataset_name, split="train")
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} rows")
    return df


def plot_accuracy(df: pd.DataFrame):
    # Calculate accuracy by dimension
    dim_acc = df.groupby("dimension_name")["reward"].mean() * 100

    # Calculate overall accuracy
    overall_acc = df["reward"].mean() * 100

    # Prepare data for plotting
    categories = ["Overall"] + list(dim_acc.index)
    accuracies = [overall_acc] + list(dim_acc.values)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        categories,
        accuracies,
        color=["#2E86AB" if cat == "Overall" else "#A23B72" for cat in categories],
    )

    # Add value labels
    for i, (cat, acc) in enumerate(zip(categories, accuracies)):
        ax.text(i, acc + 1, f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Judge Accuracy by Dimension", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("accuracy.png", dpi=300, bbox_inches="tight")
    print("\n✓ Saved: accuracy.png")
    plt.close()

    return overall_acc, dim_acc


def print_stats(overall_acc, dim_acc):
    print(f"\nOverall Accuracy: {overall_acc:.1f}%")
    print("\nAccuracy by Dimension:")
    for dim, acc in dim_acc.items():
        print(f'  {dim.replace("_", " ").title()}: {acc:.1f}%')


def main():
    df = load_perceivability_data()
    overall, dims = plot_accuracy(df)
    print_stats(overall, dims)


if __name__ == "__main__":
    main()
