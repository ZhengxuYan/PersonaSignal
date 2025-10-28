"""
Shared configuration for PersonaSignal pipeline.
Change these values to switch between different dimensions and models.
"""

import os

# HuggingFace configuration
HF_USERNAME = "JasonYan777"

# Dimension definitions
DIMENSIONS = {
    "programming_expertise": {
        "name": "programming_expertise",
        "values": ["Novice", "Intermediate", "Advanced"],
        "description": (
            "Represents the user's practical fluency in software engineering. "
            "It shapes how they decompose problems, choose abstractions, weigh tradeoffs, "
            "explain concepts, and validate solutions. Higher expertise tends to show deeper "
            "reasoning about invariants, interfaces, performance, testing strategy, and failure modes. "
            "Lower expertise favors concrete steps, worked examples, and guardrails."
        ),
    },
    "planning_horizon": {
        "name": "planning_horizon",
        "values": ["Spontaneous", "Balanced", "Strategic"],
        "description": (
            "Captures how the user sequences work and values payoff timing. "
            "Spontaneous favors immediate action, short feedback loops, and minimal upfront planning. "
            "Balanced outlines a short sequence with a checkpoint and simple contingencies. "
            "Strategic frames a long run objective with phased milestones, leading indicators, and tolerance for delayed payoff."
        ),
    },
    "locale_and_time_zone": {
        "name": "locale_and_time_zone",
        "values": [
            "US Pacific",
            "US Eastern",
            "UK",
            "EU Central",
            "India",
            "China Mainland",
            "Japan",
            "Brazil",
            "Australia",
            "Africa",
        ],
        "description": (
            "Controls the user's geographic location, timezone, date/time formatting preferences, and cultural conventions for calendar, currency, and measurements."
        ),
    },
    "verification_orientation": {
        "name": "verification_orientation",
        "values": ["Trusting", "Skeptical", "Empirical"],
        "description": (
            "Determines how much the assistant should qualify or verify its statements. "
            "Trusting users accept answers readily and prefer direct responses without excessive hedging. "
            "Skeptical users expect justification, reasoning, or cross-checking of claims. "
            "Empirical users request data sources, probabilistic framing, or evidence-based support for assertions."
        ),
    },
    "agency_expectation": {
        "name": "agency_expectation",
        "values": ["High-Agency", "Shared-Agency", "Low-Agency"],
        "description": (
            "Influences the level of initiative and decision-making the assistant should take. "
            "'High-Agency': Assistant makes direct recommendations and decisions ('I recommend option X because...'). "
            "'Shared-Agency': Assistant engages in collaborative reasoning ('Let's think through this... What do you think about...'). "
            "'Low-Agency': Assistant presents neutral information without opinions ('Here are three options: A, B, C with the following details...')."
        ),
    },
}

# Active dimension configuration
# Can be overridden by PERSONA_DIMENSION environment variable
DIMENSION_NAME = os.environ.get("PERSONA_DIMENSION", "agency_expectation")

# Model configuration
QUESTION_GEN_MODEL = "gpt-5"
PERSONA_GEN_MODEL = "gpt-4o-mini"
RESPONSE_GEN_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

# Data generation parameters
SEED = 42
NUM_QUESTIONS = 20
NUM_DISTRACTORS = 5


def get_dimension() -> dict:
    """
    Get the active dimension configuration.

    Returns:
        Dictionary with 'name', 'values', and 'description' keys
    """
    if DIMENSION_NAME not in DIMENSIONS:
        raise ValueError(
            f"Unknown dimension: {DIMENSION_NAME}. Must be one of {list(DIMENSIONS.keys())}"
        )
    return DIMENSIONS[DIMENSION_NAME]


def get_dataset_name(stage: str) -> str:
    """
    Generate HuggingFace dataset name for a given stage.

    Args:
        stage: One of 'questions', 'responses', 'perceivability'

    Returns:
        Full HuggingFace dataset name
    """
    dimension_formatted = DIMENSION_NAME.replace("_", "-").title()

    stage_names = {
        "questions": f"{HF_USERNAME}/PersonaSignal-PersonaQuestions-{dimension_formatted}",
        "responses": f"{HF_USERNAME}/PersonaSignal-PersonalizedResponse-{dimension_formatted}",
        "perceivability": f"{HF_USERNAME}/PersonaSignal-PerceivabilityTest-{dimension_formatted}",
    }

    if stage not in stage_names:
        raise ValueError(
            f"Unknown stage: {stage}. Must be one of {list(stage_names.keys())}"
        )

    return stage_names[stage]


def get_dataset_name_with_model(stage: str, model_name: str) -> str:
    """
    Generate HuggingFace dataset name for a given stage with model name.

    Args:
        stage: One of 'questions', 'responses', 'perceivability'
        model_name: Model name used in this stage (e.g., 'gpt-4o-mini')

    Returns:
        Full HuggingFace dataset name with model suffix
    """
    base_name = get_dataset_name(stage)

    if stage == "questions":
        # Questions use QUESTION_GEN_MODEL for generation and PERSONA_GEN_MODEL for personas
        return f"{base_name}-{QUESTION_GEN_MODEL}"
    elif stage == "responses":
        # Responses use RESPONSE_GEN_MODEL
        return f"{base_name}-{RESPONSE_GEN_MODEL}"
    elif stage == "perceivability":
        # Perceivability uses JUDGE_MODEL
        return f"{base_name}-{JUDGE_MODEL}"

    return base_name


def get_all_dataset_names(dimensions: list[str] = None) -> dict:
    """
    Get all dataset names for specified dimensions.

    Args:
        dimensions: List of dimension names. If None, returns for all dimensions.

    Returns:
        Dictionary mapping dimension -> stage -> dataset_name
    """
    if dimensions is None:
        dimensions = list(DIMENSIONS.keys())

    result = {}
    original_dimension = DIMENSION_NAME

    for dimension in dimensions:
        globals()["DIMENSION_NAME"] = dimension
        result[dimension] = {
            "questions": get_dataset_name("questions"),
            "responses": get_dataset_name("responses"),
            "perceivability": get_dataset_name("perceivability"),
        }

    globals()["DIMENSION_NAME"] = original_dimension
    return result
