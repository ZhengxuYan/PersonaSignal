# PersonaSignal

Pipeline for generating and evaluating personalization datasets with third-party perceivability testing.

## Overview

PersonaSignal generates personalized responses conditioned on hidden personas and evaluates whether a separate judge model can detect which persona was used based only on the response text.

## Quick Start

### 1. Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-key-here
HUGGINGFACE_TOKEN=your-token-here
```

### 2. Configure

Edit `config.py` to select a dimension:

```python
DIMENSION_NAME = "programming_expertise"  # Change this to switch dimensions
```

Available dimensions:

- `programming_expertise` - Novice, Intermediate, Advanced
- `planning_horizon` - Spontaneous, Balanced, Strategic
- `locale_and_time_zone` - Geographic and cultural contexts
- `verification_orientation` - Trusting, Skeptical, Empirical
- `agency_expectation` - High-Agency, Shared-Agency, Low-Agency

### 3. Run Pipeline

```bash
# Step 1: Generate questions and personas
python datagen/datagen.py

# Step 2: Generate personalized responses
python inference/collect_response.py

# Step 3: Test perceivability with judge
python inference/test_perceivability.py

# Step 4: Analyze results
python analyze_perceivability.py
```

## Architecture

The pipeline consists of 4 stages:

### Stage 1: Question & Persona Generation (`datagen/datagen.py`)

- Generates questions for a dimension that should elicit different responses
- Samples one value from the dimension for each question
- Creates 1 ground truth persona with sampled value + N distractor personas with different values

**Output**: `{USERNAME}/PersonaSignal-PersonaQuestions-{Dimension}`

### Stage 2: Personalized Response Generation (`inference/collect_response.py`)

- Takes the questions and personas
- Generates personalized responses using the ground truth persona
- Instructs the model to NOT explicitly mention persona traits

**Output**: `{USERNAME}/PersonaSignal-PersonalizedResponse-{Dimension}`

### Stage 3: Perceivability Testing (`inference/test_perceivability.py`)

- Presents the response and all personas (ground truth + distractors)
- Judge model must identify which persona was used
- Calculates accuracy based on correct identification

**Output**: `{USERNAME}/PersonaSignal-PerceivabilityTest-{Dimension}`

### Stage 4: Analysis (`analyze_perceivability.py`)

- Analyzes judge accuracy by dimension
- Generates visualization showing which dimensions are most detectable

**Output**: `accuracy.png`

## Combining Datasets

To combine results across multiple dimensions:

```python
# Edit combine_datasets_selective.py
dimensions_to_combine = list(config.DIMENSIONS.keys())  # All dimensions

# Run
python combine_datasets_selective.py
```

This creates:

- `PersonaSignal-All-Questions` - Combined question/persona dataset
- `PersonaSignal-All-Responses` - Combined personalized responses
- `PersonaSignal-All-Perceivability` - Combined perceivability results

## Configuration

In `config.py`, you can adjust:

```python
# Data generation
NUM_QUESTIONS = 20      # Number of questions per dimension
NUM_DISTRACTORS = 5     # Number of distractor personas

# Models
QUESTION_GEN_MODEL = "gpt-5"          # For generating questions
PERSONA_GEN_MODEL = "gpt-4o-mini"     # For generating personas
RESPONSE_GEN_MODEL = "gpt-4o-mini"   # For personalized responses
JUDGE_MODEL = "gpt-4o-mini"          # For perceivability testing

# HuggingFace
HF_USERNAME = "YourUsername"
```

## Adding a New Dimension

Add to `config.py`:

```python
DIMENSIONS = {
    # ... existing dimensions ...
    "your_new_dimension": {
        "name": "your_new_dimension",
        "values": ["Value1", "Value2", "Value3"],
        "description": "What this dimension represents...",
    },
}
```

Then set `DIMENSION_NAME = "your_new_dimension"` and run the pipeline.

## Output Format

Each dataset entry contains:

- `dimension_name` - The persona dimension
- `dimension_values` - All possible values
- `dimension_description` - What the dimension controls
- `question` - The question asked
- `why_differ` - Why responses should differ
- `how_subtle` - How personalization appears subtly
- `sampled_value` - The value used for this entry
- `ground_truth_persona` - The target persona
- `distractor_personas` - List of distractor personas
- `personalized_response` - Generated response (stages 2+)
- `judge_choice`, `judge_rationale`, `reward` - Judge results (stage 3)

## Files

```
datagen/
  ├── datagen.py          # Question & persona generation
  ├── utils.py            # LLM wrappers
  └── ...

inference/
  ├── collect_response.py  # Generate personalized responses
  └── test_perceivability.py  # Judge model evaluation

config.py                  # Central configuration
analyze_perceivability.py  # Results analysis
combine_datasets_selective.py  # Combine multiple dimensions
```

## Notes

- Personas only differ in the target dimension, with other attributes being neutral
- The assistant is instructed to NOT explicitly mention persona traits
- Judge accuracy measures how "leaky" the personalization is
- Higher accuracy = more detectable = easier to personalize (but may also indicate leakage)
