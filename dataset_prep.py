"""
Dataset Preparation Pipeline for PersonaSignal
"""

import pandas as pd
import json
from pathlib import Path
from typing import List


class DatasetPrep:
    def __init__(self, personas_file: str = "personas.csv"):
        """Initialize the dataset preparation pipeline"""
        self.personas_file = personas_file
        self.personas_df = self._load_personas()

    def _load_personas(self) -> pd.DataFrame:
        """Load personas from CSV file"""
        if Path(self.personas_file).exists():
            return pd.read_csv(self.personas_file)
        else:
            return pd.DataFrame(
                columns=[
                    "dimension",
                    "dimension_value",
                    "prompt",
                    "target_persona",
                    "distractor_personas",
                    "possible_dimension_values",
                ]
            )

    def _save_personas(self):
        """Save personas to CSV file"""
        self.personas_df.to_csv(self.personas_file, index=False)

    def add_persona(
        self,
        dimension: str,
        possible_dimension_values: List[str],
        dimension_value: str,
        prompt: str,
        target_persona: str,
        distractor_personas: List[str],
    ):
        """
        Add a persona entry

        Args:
            dimension: The dimension name (e.g., "locale_timezone")
            possible_dimension_values: List of all possible values for this dimension
            dimension_value: The value for this dimension (e.g., "US Pacific")
            prompt: The question for this entry
            target_persona: Full persona description with this dimension value
            distractor_personas: List of distractor personas with different dimension values
        """
        if len(distractor_personas) < 1:
            raise ValueError("Must provide at least 1 distractor persona")

        # Convert lists to pipe-separated strings for storage
        values_str = "|".join(possible_dimension_values)
        distractors_str = "|".join(distractor_personas)

        new_row = pd.DataFrame(
            [
                {
                    "dimension": dimension,
                    "dimension_value": dimension_value,
                    "prompt": prompt,
                    "target_persona": target_persona,
                    "distractor_personas": distractors_str,
                    "possible_dimension_values": values_str,
                }
            ]
        )

        self.personas_df = pd.concat([self.personas_df, new_row], ignore_index=True)
        self._save_personas()
        print(f"Added persona for {dimension}={dimension_value}")

    def get_all_personas_for_dimension(self, dimension: str) -> pd.DataFrame:
        """Get all persona entries for a specific dimension"""
        return self.personas_df[self.personas_df["dimension"] == dimension]

    def create_dataset(self, output_file: str = "final_dataset.json"):
        """
        Create dataset from personas with their associated prompts

        Args:
            output_file: Output JSON file
        """
        dataset = []

        for idx, row in self.personas_df.iterrows():
            entry = {
                "question_id": f"{row['dimension']}_q{idx+1}",
                "dimension": row["dimension"],
                "dimension_value": row["dimension_value"],
                "question": row["prompt"],
                "target_persona": row["target_persona"],
                "distractor_personas": row["distractor_personas"].split("|"),
                "possible_dimension_values": row["possible_dimension_values"].split(
                    "|"
                ),
            }
            dataset.append(entry)

        with open(output_file, "w") as f:
            json.dump(dataset, f, indent=2)

        print(f"\nDataset created: {output_file}")
        print(f"Total entries: {len(dataset)}")

        # Show breakdown by dimension
        dimension_counts = {}
        for entry in dataset:
            dim = entry["dimension"]
            dimension_counts[dim] = dimension_counts.get(dim, 0) + 1

        print("\nBreakdown by dimension:")
        for dim, count in dimension_counts.items():
            print(f"  {dim}: {count} questions")

        return dataset

    def list_personas(self):
        """Display all personas"""
        print("\n=== Personas in Dataset ===\n")
        if self.personas_df.empty:
            print("No personas found. Add personas using add_persona().")
            return

        for dimension in self.personas_df["dimension"].unique():
            dimension_personas = self.get_all_personas_for_dimension(dimension)
            print(f"Dimension: {dimension}")
            print(f"  Entries: {len(dimension_personas)}")
            print()


def main():
    import sys

    prep = DatasetPrep()

    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        prep.create_dataset()
    else:
        prep.list_personas()


if __name__ == "__main__":
    main()
