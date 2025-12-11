import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_win_rates(input_file="comparison_results.jsonl", output_file="win_rates_by_dimension.png"):
    """
    Generate a publication-quality bar plot of win rates per dimension.
    """
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    print(f"Loading results from {input_file}...")
    df = pd.read_json(input_file, lines=True)

    # 1. Prepare Data
    # Calculate win counts per dimension and model
    # Rename model for better presentation
    df["winner_model"] = df["winner_model"].replace("DPO Tinker", "DPO Trained Llama 3.1 8B")
    
    # Rename model for better presentation
    df["winner_model"] = df["winner_model"].replace("DPO Tinker", "DPO Trained Llama 3.1 8B")
    
    # 1a. Per-Dimension Win Rates
    counts = df.groupby(["dimension_name", "winner_model"]).size().reset_index(name="wins")
    totals = df.groupby("dimension_name").size().reset_index(name="total")
    merged = pd.merge(counts, totals, on="dimension_name")
    merged["win_rate"] = merged["wins"] / merged["total"]

    # 1b. Overall Win Rate
    overall_counts = df.groupby("winner_model").size().reset_index(name="wins")
    overall_total = len(df)
    overall_counts["total"] = overall_total
    overall_counts["win_rate"] = overall_counts["wins"] / overall_total
    overall_counts["dimension_name"] = "Overall"

    # Combine
    merged = pd.concat([merged, overall_counts], ignore_index=True)

    # 2. Setup Aesthetic Style
    sns.set_theme(style="white", context="paper", font_scale=1.4)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)

    # 3. Create Bar Plot
    models = sorted(df["winner_model"].unique())
    
    # Ensure "Overall" is at the end or beginning? Let's put it at the end.
    # Get unique dimensions sorted, but keep Overall last.
    dims = sorted([d for d in merged["dimension_name"].unique() if d != "Overall"])
    dims.append("Overall")
    
    bar_plot = sns.barplot(
        data=merged,
        x="dimension_name",
        y="win_rate",
        hue="winner_model",
        hue_order=models,
        order=dims,
        palette="Paired", 
        edgecolor="black",
        linewidth=1.0,
        ax=ax
    )

    # 4. Refine Axes and Labels
    ax.set_title("Win Rates by Dimension", fontsize=20, weight='bold', pad=30)
    ax.text(0.5, 1.02, "Judge: claude-sonnet-4-5-20250929", transform=ax.transAxes, 
            ha='center', fontsize=14, color='#555555')
    
    ax.set_ylabel("Win Rate", fontsize=16, weight='bold')
    ax.set_xlabel("Dimension", fontsize=16, weight='bold', labelpad=15)
    
    # Format Y-axis as percentage and increase limit to fit labels
    ax.set_ylim(0, 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    # Rotate X-axis labels for readability
    xticks = ax.get_xticklabels()
    ax.set_xticklabels(xticks, rotation=45, ha="right", fontsize=12)

    # 5. Add Percentage Labels on Bars
    for container in ax.containers:
        # Custom label formatting to ensure it's readable
        labels = [f'{v.get_height():.1%}' for v in container]
        ax.bar_label(container, labels=labels, padding=3, fontsize=11, weight='bold')

    # 6. Customize Legend
    ax.legend(
        title="Model", 
        title_fontsize=14, 
        fontsize=12, 
        loc="upper left", 
        bbox_to_anchor=(1, 1),
        frameon=True,
        shadow=True
    )

    # 7. Clean Layout
    sns.despine(left=False, bottom=False)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # 8. Save
    print(f"Saving plot to {output_file}...")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
    print("Done!")

if __name__ == "__main__":
    plot_win_rates()
