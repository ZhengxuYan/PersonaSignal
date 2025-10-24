# PersonaSignal

Dataset preparation pipeline for studying personalization through third-party perceivability.

## Setup

```bash
pip install -r requirements.txt
```

## Structure

**personas.csv**: Each row contains a complete entry with dimension, prompt, and personas

- `dimension`: The dimension name (e.g., "locale_timezone")
- `dimension_value`: The value for this entry (e.g., "US Pacific")
- `prompt`: The question for this entry
- `target_persona`: Full persona description with this dimension value
- `distractor_personas`: List of distractor personas with different dimension values (pipe-separated)
- `possible_dimension_values`: All possible values for this dimension (pipe-separated)

## Usage

### 1. Add entries to personas.csv

```python
from datagen.dataset_prep import DatasetPrep

prep = DatasetPrep()

# Add a complete entry with dimension, prompt, and personas
prep.add_persona(
    dimension="locale_timezone",
    possible_dimension_values=["US Pacific", "US Eastern", "UK", "EU Central", "India", "China Mainland", "Japan", "Brazil", "Australia"],
    dimension_value="US Pacific",
    prompt="What time zone are you in?",
    target_persona="Sarah is a 32-year-old software engineer living in San Francisco. She works in tech and enjoys hiking on weekends.",
    distractor_personas=[
        "James is a 28-year-old teacher in London. He enjoys reading and theater.",
        "Yuki is a 35-year-old designer in Tokyo. She loves anime and photography.",
        "Raj is a 40-year-old doctor in Mumbai. He enjoys cricket and cooking."
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
  "question_id": "locale_timezone_q1",
  "dimension": "locale_timezone",
  "dimension_value": "US Pacific",
  "question": "What time zone are you in?",
  "target_persona": "Sarah is a 32-year-old...",
  "distractor_personas": [
    "James is a 28-year-old...",
    "Yuki is a 35-year-old...",
    "Raj is a 40-year-old..."
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

1. For each dimension, identify the possible values
2. For each question you want to ask:
   - Choose a dimension value for the question
   - Use an LLM to generate a target persona with that dimension value
   - Use an LLM to generate distractor personas with different dimension values
   - Include filler attributes (age, sex, job, hobbies) that don't affect answers
3. Add each entry to `datagen/data_with_personas.csv` using `add_persona()`
4. Run `python datagen/dataset_prep.py generate` to create the final dataset
5. Use the dataset to generate responses and evaluate perceivability

## Notes

- Each persona should include filler information (age, sex, job, etc.) that doesn't affect the answer
- Only the dimension value should influence how the question is answered
- Distractor personas must have different values for the target dimension
