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
                    "possible_dimension_values",
                    "dimension_value",
                    "target_persona",
                    "distractor_personas",
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
        target_persona: str,
        distractor_personas: List[str],
    ):
        """
        Add a persona entry

        Args:
            dimension: The dimension name (e.g., "locale_timezone")
            possible_dimension_values: List of all possible values for this dimension
            dimension_value: The value for this dimension (e.g., "US Pacific")
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
                    "possible_dimension_values": values_str,
                    "dimension_value": dimension_value,
                    "target_persona": target_persona,
                    "distractor_personas": distractors_str,
                }
            ]
        )

        self.personas_df = pd.concat([self.personas_df, new_row], ignore_index=True)
        self._save_personas()
        print(f"Added persona for {dimension}={dimension_value}")

    def get_all_personas_for_dimension(self, dimension: str) -> pd.DataFrame:
        """Get all persona entries for a specific dimension"""
        return self.personas_df[self.personas_df["dimension"] == dimension]

    def create_dataset_from_prompts(
        self, prompts_file: str, output_file: str = "final_dataset.json"
    ):
        """
        Create dataset by pairing personas with prompts

        Args:
            prompts_file: JSON file with prompts per dimension
            output_file: Output JSON file

        Expected prompts_file format:
        {
          "dimension_name": ["prompt1", "prompt2", ...],
          ...
        }
        """
        with open(prompts_file, "r") as f:
            prompts = json.load(f)

        dataset = []

        for dimension, dimension_prompts in prompts.items():
            dimension_personas = self.get_all_personas_for_dimension(dimension)

            if dimension_personas.empty:
                print(f"Warning: No personas found for dimension '{dimension}'")
                continue

            for i, prompt in enumerate(dimension_prompts, 1):
                # Randomly select a persona entry for this question
                persona_row = dimension_personas.sample(n=1).iloc[0]

                entry = {
                    "question_id": f"{dimension}_q{i}",
                    "dimension": dimension,
                    "possible_dimension_values": persona_row[
                        "possible_dimension_values"
                    ].split("|"),
                    "dimension_value": persona_row["dimension_value"],
                    "question": prompt,
                    "target_persona": persona_row["target_persona"],
                    "distractor_personas": persona_row["distractor_personas"].split(
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
        if len(sys.argv) < 3:
            print("Usage: python dataset_prep.py generate <prompts.json>")
            sys.exit(1)

        prompts_file = sys.argv[2]
        prep.create_dataset_from_prompts(prompts_file)
    else:
        prep.list_personas()


if __name__ == "__main__":
    main()
