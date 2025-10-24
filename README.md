# PersonaSignal

Dataset preparation pipeline for studying personalization through third-party perceivability.

## Setup

```bash
pip install -r requirements.txt
```

## Structure

**datagen/data_with_personas.csv**: Each row contains a complete entry with dimension, question, and personas

- `dimension_name`: The dimension name (e.g., "locale_and_time_zone")
- `dimension_values`: List of all possible values for this dimension (NumPy array format)
- `dimension_description`: Description of what this dimension controls
- `question`: The question/prompt for this entry
- `why_differ`: Explanation of why responses differ across dimension values
- `how_subtle`: Explanation of how personalization cues appear subtly
- `sampled_value`: The specific dimension value for this entry (e.g., "US Pacific")
- `num_distractors`: Number of distractor personas
- `ground_truth_persona`: The target persona description with the sampled dimension value
- `distractor_personas`: List of distractor personas with different dimension values (NumPy array format)

## Usage

### 1. Add entries to data_with_personas.csv

```python
from datagen.dataset_prep import DatasetPrep

prep = DatasetPrep()

# Add a complete entry with dimension, question, and personas
prep.add_persona(
    dimension_name="locale_and_time_zone",
    dimension_values=["US Pacific", "US Eastern", "UK", "EU Central", "India", "China Mainland", "Japan", "Brazil", "Australia"],
    dimension_description="Controls the user's geographic location, timezone, date/time formatting preferences, and cultural conventions.",
    question="Could you suggest two time slots for a weekly meeting?",
    why_differ="Different locales will bias toward morning or late-day local options and reflect local time notation.",
    how_subtle="Cues show up as which hours are proposed and use of 12-hour vs 24-hour and day-first vs month-first dates.",
    sampled_value="US Pacific",
    num_distractors=5,
    ground_truth_persona="A project coordinator based in San Francisco who plans meetings during standard PST office hours. Uses 12-hour clock and MM/DD date format.",
    distractor_personas=[
        "A project coordinator in New York using EST, 12-hour clock, and MM/DD dates.",
        "A coordinator in London using GMT/BST, 24-hour clock, and DD/MM dates.",
        "A coordinator in Berlin using CET, 24-hour clock, and DD.MM dates.",
        "A coordinator in Mumbai using IST, 24-hour clock, and DD-MM dates.",
        "A coordinator in Tokyo using JST, 24-hour clock, and YYYY/MM/DD dates."
    ]
)
```

### 2. View current entries

```bash
python datagen/dataset_prep.py
```

### 3. Generate final dataset

```bash
python datagen/dataset_prep.py generate
```

This creates `datagen/final_dataset.json` with entries like:

```json
{
  "question_id": "locale_and_time_zone_q1",
  "dimension": "locale_and_time_zone",
  "dimension_value": "US Pacific",
  "question": "Could you suggest two time slots for a weekly meeting?",
  "target_persona": "A project coordinator based in San Francisco...",
  "distractor_personas": [
    "A project coordinator in New York using EST...",
    "A coordinator in London using GMT/BST...",
    "A coordinator in Berlin using CET...",
    "A coordinator in Mumbai using IST...",
    "A coordinator in Tokyo using JST..."
  ],
  "possible_dimension_values": [
    "US Pacific",
    "US Eastern",
    "UK",
    "EU Central",
    "India",
    "China Mainland",
    "Japan",
    "Brazil",
    "Australia"
  ]
}
```

## Workflow

1. Define a dimension with its possible values and description
2. For each question you want to ask:
   - Choose a dimension value (sampled_value) for the question
   - Write why responses should differ (why_differ) and how personalization appears subtly (how_subtle)
   - Use an LLM to generate a ground truth persona with that dimension value
   - Use an LLM to generate distractor personas with different dimension values
   - Include filler attributes (age, sex, job, hobbies) that don't affect answers
3. Add entries to `datagen/data_with_personas.csv` using `add_persona()` or directly edit the CSV
4. Run `python datagen/dataset_prep.py generate` to create `datagen/final_dataset.json`
5. Use the dataset to generate responses and evaluate perceivability with an LLM judge

## Notes

- Each persona should include filler information (age, sex, job, etc.) that doesn't affect the answer
- Only the dimension value should influence how the question is answered
- Distractor personas must have different values for the target dimension
- The parser handles both Python list format (`['a', 'b']`) and NumPy array format (`['a' 'b']`)
