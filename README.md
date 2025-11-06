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

Available dimensions (see `config.py` for full details):

- `programming_expertise` - Novice, Intermediate, Advanced
- `planning_horizon` - Spontaneous, Balanced, Strategic
- `locale_and_time_zone` - Geographic and cultural contexts
- `verification_orientation` - Trusting, Skeptical, Empirical
- `agency_expectation` - High-Agency, Shared-Agency, Low-Agency
- `communication_formality` - Formal, Neutral, Casual
- `exploration_tendency` - Conservative, Moderate, Exploratory
- `social_scope` - Individual-Focused, Balanced, Community-Oriented
- `learning_goal` - Mastery, Practical Application, Theoretical Understanding
- `feedback_style` - Direct, Balanced, Gentle

### 3. Run Pipeline

#### Option A: Run Complete Pipeline (Recommended)

Run all stages for multiple dimensions at once:

```bash
# Run all stages for specific dimensions
python run_all_dimensions.py -d programming_expertise planning_horizon

# Run specific stages only
python run_all_dimensions.py -d programming_expertise -s datagen collect_response

# Skip certain stages
python run_all_dimensions.py -d programming_expertise --skip-combine --skip-analysis

# Run all dimensions (uses all dimensions from config.py)
python run_all_dimensions.py
```

Available stages:
- `datagen` - Generate questions and personas
- `collect_response` - Generate personalized responses
- `test_perceivability` - Run judge evaluation
- Dataset combination (automatic unless `--skip-combine`)
- Analysis (automatic unless `--skip-analysis`)

#### Option B: Run Individual Scripts

Run each stage manually for fine-grained control:

```bash
# Step 1: Generate questions and personas
python datagen/datagen.py

# Step 2: Generate personalized responses
python inference/collect_response.py

# Step 3: Test perceivability with judge
python inference/test_perceivability.py

# Step 4: Combine datasets (optional)
python combine_datasets_selective.py

# Step 5: Analyze results
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

## Common Usage Patterns

### Generate Full Dataset for Multiple Dimensions

```bash
# Generate everything for specific dimensions
python run_all_dimensions.py -d programming_expertise planning_horizon locale_and_time_zone

# Generate for all dimensions in config
python run_all_dimensions.py
```

### Re-run Specific Stages

```bash
# Re-generate responses with a different model (update RESPONSE_GEN_MODEL in config.py first)
python run_all_dimensions.py -d programming_expertise -s collect_response test_perceivability

# Re-run just the judge evaluation
python run_all_dimensions.py -d programming_expertise -s test_perceivability
```

### Compare Multiple Models

```bash
# 1. Generate responses with first model
# (Set RESPONSE_GEN_MODEL = "gpt-4o-mini" in config.py)
python run_all_dimensions.py -d programming_expertise

# 2. Generate responses with second model
# (Set RESPONSE_GEN_MODEL = "gpt-4o" in config.py)
python run_all_dimensions.py -d programming_expertise -s collect_response test_perceivability

# 3. Compare results
# (Edit analyze_perceivability.py to compare both models)
python analyze_perceivability.py
```

### Combining Datasets

The `run_all_dimensions.py` script automatically combines datasets unless you use `--skip-combine`. To manually combine:

```bash
python combine_datasets_selective.py
```

This creates:
- `PersonaSignal-All-Questions-{model}` - Combined question/persona dataset
- `PersonaSignal-All-Responses-{model}` - Combined personalized responses
- `PersonaSignal-All-Perceivability-{model}` - Combined perceivability results

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

## Scripts Reference

### Main Pipeline Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_all_dimensions.py` | Orchestrates entire pipeline for multiple dimensions | `python run_all_dimensions.py -d dimension1 dimension2` |
| `datagen/datagen.py` | Generate questions and personas for one dimension | `python datagen/datagen.py` |
| `inference/collect_response.py` | Generate personalized responses | `python inference/collect_response.py` |
| `inference/test_perceivability.py` | Run judge evaluation | `python inference/test_perceivability.py` |
| `combine_datasets_selective.py` | Combine datasets across dimensions | `python combine_datasets_selective.py` |
| `analyze_perceivability.py` | Generate accuracy plots and statistics | `python analyze_perceivability.py` |

### Configuration Files

| File | Purpose |
|------|---------|
| `config.py` | Central configuration for models, dimensions, HuggingFace username |
| `.env` | API keys and tokens (create this yourself) |
| `requirements.txt` | Python dependencies |

### Project Structure

```
datagen/
  ├── datagen.py          # Question & persona generation
  └── ...

inference/
  ├── collect_response.py  # Generate personalized responses
  └── test_perceivability.py  # Judge model evaluation

config.py                  # Central configuration
run_all_dimensions.py      # Main pipeline orchestrator
analyze_perceivability.py  # Results analysis and visualization
combine_datasets_selective.py  # Combine multiple dimensions
```

## Notes

- Personas only differ in the target dimension, with other attributes being neutral
- The assistant is instructed to NOT explicitly mention persona traits
- Judge accuracy measures how "leaky" the personalization is
- Higher accuracy = more detectable = easier to personalize (but may also indicate leakage)
