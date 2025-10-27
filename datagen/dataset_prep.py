"""
Dataset Preparation Pipeline for PersonaSignal
"""

import pandas as pd
import json
import ast
import re
from pathlib import Path
from typing import List


class DatasetPrep:
    def __init__(self, personas_file: str = "datagen/data_with_personas.csv"):
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
                    "dimension_name",
                    "dimension_values",
                    "dimension_description",
                    "question",
                    "why_differ",
                    "how_subtle",
                    "sampled_value",
                    "num_distractors",
                    "ground_truth_persona",
                    "distractor_personas",
                ]
            )

    def _save_personas(self):
        """Save personas to CSV file"""
        self.personas_df.to_csv(self.personas_file, index=False)

    def _parse_array_string(self, s):
        """Parse string representation of array to actual list (handles both Python and NumPy formats)"""
        if pd.isna(s):
            return []
        if isinstance(s, list):
            return s

        if not isinstance(s, str):
            return [str(s)]

        s = s.strip()

        # Check if it looks like NumPy array format (space-separated quoted strings)
        # NumPy format: ['item1' 'item2'] - no commas between items
        # Python format: ['item1', 'item2'] - has commas
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1]
            # Check if it's NumPy format (has quotes but few/no commas)
            # Count quotes and commas to detect format
            single_quotes = inner.count("'")
            commas = inner.count(",")

            # If we have multiple quoted strings but few commas, it's likely NumPy format
            if single_quotes >= 4 and commas < (single_quotes / 4):
                # NumPy format detected - parse quoted strings
                matches = re.findall(r"'([^']*)'", inner)
                if not matches:
                    matches = re.findall(r'"([^"]*)"', inner)
                if matches:
                    return matches

        # Try to parse as Python literal (comma-separated)
        try:
            result = ast.literal_eval(s)
            if isinstance(result, list):
                return result
        except (ValueError, SyntaxError):
            pass

        # Fallback: return as single item
        return [s] if s else []

    def add_persona(
        self,
        dimension_name: str,
        dimension_values: List[str],
        dimension_description: str,
        question: str,
        why_differ: str,
        how_subtle: str,
        sampled_value: str,
        num_distractors: int,
        ground_truth_persona: str,
        distractor_personas: List[str],
    ):
        """
        Add a persona entry

        Args:
            dimension_name: The dimension name (e.g., "locale_and_time_zone")
            dimension_values: List of all possible values for this dimension
            dimension_description: Description of the dimension
            question: The question for this entry
            why_differ: Explanation of why responses differ
            how_subtle: Explanation of how differences are subtle
            sampled_value: The sampled dimension value for this entry
            num_distractors: Number of distractor personas
            ground_truth_persona: The target persona description
            distractor_personas: List of distractor persona descriptions
        """
        new_row = pd.DataFrame(
            [
                {
                    "dimension_name": dimension_name,
                    "dimension_values": str(dimension_values),
                    "dimension_description": dimension_description,
                    "question": question,
                    "why_differ": why_differ,
                    "how_subtle": how_subtle,
                    "sampled_value": sampled_value,
                    "num_distractors": num_distractors,
                    "ground_truth_persona": ground_truth_persona,
                    "distractor_personas": str(distractor_personas),
                }
            ]
        )

        self.personas_df = pd.concat([self.personas_df, new_row], ignore_index=True)
        self._save_personas()
        print(f"Added persona for {dimension_name}={sampled_value}")

    def get_all_personas_for_dimension(self, dimension_name: str) -> pd.DataFrame:
        """Get all persona entries for a specific dimension"""
        return self.personas_df[self.personas_df["dimension_name"] == dimension_name]

    def create_dataset(self, output_file: str = None):
        """
        Create dataset from personas with their associated prompts

        Args:
            output_file: Output JSON file (defaults to final_dataset.json in same directory as script)
        """
        if output_file is None:
            script_dir = Path(__file__).parent
            output_file = str(script_dir / "final_dataset.json")

        dataset = []

        for idx, row in self.personas_df.iterrows():
            # Parse array strings
            dimension_values = self._parse_array_string(row["dimension_values"])
            distractor_personas = self._parse_array_string(row["distractor_personas"])

            entry = {
                "question_id": f"{row['dimension_name']}_q{idx+1}",
                "dimension": row["dimension_name"],
                "dimension_value": row["sampled_value"],
                "question": row["question"],
                "target_persona": row["ground_truth_persona"],
                "distractor_personas": distractor_personas,
                "possible_dimension_values": dimension_values,
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

        for dimension in self.personas_df["dimension_name"].unique():
            dimension_personas = self.get_all_personas_for_dimension(dimension)
            print(f"Dimension: {dimension}")
            print(f"  Entries: {len(dimension_personas)}")

            # Show first entry details
            if len(dimension_personas) > 0:
                first = dimension_personas.iloc[0]
                print(f"  Description: {first['dimension_description'][:80]}...")
                values = self._parse_array_string(first["dimension_values"])
                print(
                    f"  Values: {', '.join(values[:5])}{'...' if len(values) > 5 else ''}"
                )
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
