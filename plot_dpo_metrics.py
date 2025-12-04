import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def plot_metrics(jsonl_path, output_path="dpo_training_plots.png"):
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not data:
        print("No data found in the metrics file.")
        return

    df = pd.DataFrame(data)
    
    # Filter for relevant columns
    metrics_to_plot = ['dpo_loss', 'accuracy', 'margin', 'chosen_reward', 'rejected_reward']
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    if not available_metrics:
        print("No relevant metrics found to plot.")
        return

    # Create subplots
    num_metrics = len(available_metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics), sharex=True)
    
    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        ax.plot(df['step'], df[metric], label=metric)
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} over steps')
        ax.grid(True)
        ax.legend()
        
        # Add moving average for smoother visualization
        window = min(len(df), 50)
        if window > 1:
            df[f'{metric}_ma'] = df[metric].rolling(window=window).mean()
            ax.plot(df['step'], df[f'{metric}_ma'], label=f'{metric} (MA {window})', linestyle='--', alpha=0.7)
            ax.legend()

    axes[-1].set_xlabel('Step')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plots saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_dpo_metrics.py <path_to_metrics.jsonl> [output_image_path]")
        sys.exit(1)
        
    jsonl_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "dpo_training_plots.png"
    
    if not os.path.exists(jsonl_file):
        print(f"Error: File {jsonl_file} not found.")
        sys.exit(1)
        
    plot_metrics(jsonl_file, output_file)
