from datasets import load_dataset
import pandas as pd

# Load the dataset
dataset_name = "JasonYan777/PersonaSignal-All-Perceivability-gpt-4o"
print(f"Loading dataset: {dataset_name}")
ds = load_dataset(dataset_name)

# Convert to pandas DataFrame
df = ds['train'].to_pandas()

# Group by 'dimension_name' and sample 10 rows
sampled_df = df.groupby('dimension_name').apply(lambda x: x.sample(n=min(len(x), 10), random_state=42)).reset_index(drop=True)

# Print stats
print("\nSampling complete.")
print("Counts per dimension:")
print(sampled_df['dimension_name'].value_counts())

# Save to JSONL
output_file = "sampled_dataset.jsonl"
sampled_df.to_json(output_file, orient='records', lines=True)
print(f"\nSaved sampled dataset to {output_file}")
