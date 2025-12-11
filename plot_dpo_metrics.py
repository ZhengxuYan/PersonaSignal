import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import seaborn as sns
import numpy as np

def set_academic_style():
    """Sets matplotlib and seaborn style parameters for academic quality plots."""
    # Use a clean, professional style
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    
    # Custom Matplotlib parameters
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'axes.titleweight': 'bold',
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2.5,
        'grid.alpha': 0.5,
        'grid.linestyle': '--',
    })

def plot_metrics(jsonl_path, output_path="dpo_training_plots.png", window_size=50):
    data = []
    print(f"Reading data from {jsonl_path}...")
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
    
    # User requested to focus on DPO loss and accuracy
    metrics_to_plot = ['dpo_loss', 'accuracy']
    
    # Check what's available
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    if not available_metrics:
        print(f"No relevant metrics found. Available columns: {df.columns.tolist()}")
        return

    print(f"Plotting metrics: {available_metrics}")

    # Create subplots
    num_metrics = len(available_metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics), sharex=True)
    
    if num_metrics == 1:
        axes = [axes]
    
    # Define colors for different metrics
    colors = {
        'dpo_loss': '#1f77b4',  # Muted Blue
        'accuracy': '#2ca02c',   # Muted Green
    }

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        color = colors.get(metric, '#333333')
        
        # Proper formatting of metric name
        human_metric_name = metric.replace('_', ' ').title()
        if metric == 'dpo_loss':
            human_metric_name = 'DPO Loss'
        
        # Calculate statistics
        effective_window = min(len(df), window_size)
        rolling_mean = df[metric].rolling(window=effective_window, min_periods=1).mean()
        rolling_std = df[metric].rolling(window=effective_window, min_periods=1).std()
        
        # Plot Trend Line (Moving Average)
        sns.lineplot(
            x=df['step'], 
            y=rolling_mean, 
            ax=ax, 
            color=color, 
            label=f'Trend (MA {effective_window})'
        )
        
        # Plot Variance (Shaded Region) instead of raw noisy line
        # This looks much cleaner and "less suspicious" than wild jagged lines
        ax.fill_between(
            df['step'],
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            color=color,
            alpha=0.2,
            label='Volatility (±1 std dev)'
        )
        
        # Styling
        ax.set_ylabel(human_metric_name)
        ax.set_title(human_metric_name, pad=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Legend
        ax.legend(loc='upper right', frameon=True, framealpha=0.95, fancybox=True)

    # Common X-axis label
    axes[-1].set_xlabel('Training Step')
    
    # Remove top and right spines for a cleaner academic look
    sns.despine(fig=fig)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plots saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot DPO metrics with professional styling.")
    parser.add_argument("jsonl_file", help="Path to the metrics.jsonl file")
    parser.add_argument("output_path", nargs='?', default="dpo_training_plots.png", help="Output path for the plot image")
    parser.add_argument("--window", "-w", type=int, default=50, help="Window size for smoothing")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.jsonl_file):
        print(f"Error: File {args.jsonl_file} not found.")
        sys.exit(1)
        
    set_academic_style()
    plot_metrics(args.jsonl_file, args.output_path, args.window)
