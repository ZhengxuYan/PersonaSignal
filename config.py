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
    "communication_formality": {
        "name": "communication_formality",
        "values": ["Casual", "Professional", "Formal"],
        "description": (
            "Controls the tone and register of communication. "
            "Casual users prefer conversational language with contractions and friendly expressions. "
            "Professional users expect polished, business-appropriate language without being stiff. "
            "Formal users require precise, grammatically rigorous language with complete sentences and professional terminology."
        ),
    },
    "exploration_tendency": {
        "name": "exploration_tendency",
        "values": ["Conservative", "Pragmatic", "Exploratory"],
        "description": (
            "Reflects the user's openness to novel versus proven approaches. "
            "Conservative users prefer well-established methods with track records and minimal risk. "
            "Pragmatic users balance reliability with selective innovation, open to new ideas with reasonable validation. "
            "Exploratory users embrace novelty, experimentation, and cutting-edge approaches even with uncertainty."
        ),
    },
    "social_scope": {
        "name": "social_scope",
        "values": ["Individual", "Relational", "Collective"],
        "description": (
            "Determines whose interests and impacts the assistant should consider in framing advice. "
            "Individual users optimize purely for personal goals, efficiency, and self-interest. "
            "Relational users consider immediate social circle—how decisions affect family, friends, or close colleagues. "
            "Collective users frame problems in terms of community welfare, organizational impact, or broader societal consequences."
        ),
    },
    "learning_goal": {
        "name": "learning_goal",
        "values": ["Solution-Focused", "Skill-Building"],
        "description": (
            "Distinguishes between solving the immediate problem versus building lasting capability. "
            "Solution-Focused users want the fastest path to completion—direct answers, ready-to-use code, or concrete steps. "
            "Skill-Building users prefer brief explanations of underlying principles alongside the solution, "
            "highlighting key patterns that transfer to similar problems."
        ),
    },
    "feedback_style": {
        "name": "feedback_style",
        "values": ["Directive", "Guided", "Socratic"],
        "description": (
            "Controls how the assistant delivers corrections, explanations, and learning feedback. "
            "Directive users prefer clear, explicit instruction—'Here's what's wrong and here's the fix.' "
            "Guided users want structured hints and reasoning—'Consider this aspect... What if you tried...?' "
            "Socratic users benefit from questions that prompt self-discovery—'Why might this approach fail? What pattern do you notice?'"
        ),
    },
}

# Active dimension configuration
# Can be overridden by PERSONA_DIMENSION environment variable
DIMENSION_NAME = os.environ.get("PERSONA_DIMENSION", "agency_expectation")

# Model configuration
QUESTION_GEN_MODEL = "DPO"
PERSONA_GEN_MODEL = "gpt-4o-mini"
RESPONSE_GEN_MODEL = "DPO-Tinker"
JUDGE_MODEL = "gpt-5-mini"

# Backend configuration
# Can be overridden by BACKEND environment variable
# Supported backends: "litellm", "openai", "anthropic", etc.
# Set to None to use the default backend
BACKEND = os.environ.get("BACKEND", "litellm")
# BACKEND = None  # Uncomment to use default backend

# Backend parameters (e.g., rate limits)
# Set to None to use default backend settings
BACKEND_PARAMS = {
    "max_requests_per_minute": 2_000,  # 2K requests/minute
    "max_tokens_per_minute": 4_000_000,  # 4M tokens/minute
}
# BACKEND_PARAMS = None  # Uncomment to disable custom backend params

# Data generation parameters
SEED = 42
NUM_QUESTIONS = 50
NUM_DISTRACTORS = 5

# Dataset mode: append to existing dataset or overwrite
# Can be overridden by APPEND_DATASET environment variable (set to "true" to append)
APPEND_MODE = os.environ.get("APPEND_DATASET", "false").lower() == "true"


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
        "leakage_check": f"{HF_USERNAME}/PersonaSignal-LeakageCheck-{dimension_formatted}",
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
        return f"{base_name}-{QUESTION_GEN_MODEL.split('/')[-1]}"
    elif stage == "responses":
        # Responses use RESPONSE_GEN_MODEL
        return f"{base_name}-{RESPONSE_GEN_MODEL.split('/')[-1]}"
    elif stage == "perceivability":
        # Perceivability datasets are keyed by RESPONSE_GEN_MODEL (the model being evaluated)
        # not JUDGE_MODEL (the evaluation tool)
        return f"{base_name}-{RESPONSE_GEN_MODEL.split('/')[-1]}"
    elif stage == "leakage_check":
        # Leakage check datasets are also keyed by RESPONSE_GEN_MODEL
        return f"{base_name}-{RESPONSE_GEN_MODEL.split('/')[-1]}"

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
