import sys
import os
from pathlib import Path
import random
from datasets import load_dataset, concatenate_datasets
from dotenv import load_dotenv

# Add parent directory to path to import config and inference modules
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from inference.collect_response import PersonalizedResponseGenerator
from inference.test_perceivability import PerceivabilityTestGenerator

load_dotenv()

def generate_all_dpo_pairs(
    num_responses_per_question: int = 10,
    response_model: str = None,
    judge_model: str = None
):
    if response_model is None:
        response_model = config.RESPONSE_GEN_MODEL
    if judge_model is None:
        judge_model = config.JUDGE_MODEL
        
    all_dpo_pairs = []
    
    for dimension_name in config.DIMENSIONS.keys():
        print(f"\n=== Processing Dimension: {dimension_name} ===")
        
        # 1. Load Questions Dataset
        try:
            # We construct the dataset name manually or via config helper if it supports dimension arg
            # config.get_dataset_name_with_model uses config.DIMENSION_NAME by default, 
            # so we need to temporarily set it or manually construct the name.
            # Looking at config.py (not fully shown but assuming pattern), let's manually construct it to be safe
            # or check if get_dataset_name_with_model accepts dimension override.
            # The previous code used: config.get_dataset_name_with_model("questions", config.QUESTION_GEN_MODEL)
            # which relies on config.DIMENSION_NAME.
            
            # Let's assume the naming convention: 
            # f"{config.HF_USERNAME}/PersonaSignal-PersonaQuestions-{dimension_name_formatted}-{model_suffix}"
            # But safer to just temporarily patch config.DIMENSION_NAME or use the helper if I can see it.
            # I'll manually construct it based on the previous log output: 
            # "JasonYan777/PersonaSignal-PersonaQuestions-Agency-Expectation-gpt-5"
            
            # Helper to format dimension name (title case, hyphens)
            formatted_dim = dimension_name.replace("_", "-").title()
            input_dataset_name = f"{config.HF_USERNAME}/PersonaSignal-PersonaQuestions-{formatted_dim}-{config.QUESTION_GEN_MODEL}"
            
            print(f"Loading questions from: {input_dataset_name}")
            dataset = load_dataset(input_dataset_name, split="train")
        except Exception as e:
            print(f"Skipping dimension {dimension_name} due to error loading dataset: {e}")
            continue
        
        # 2. Expand Dataset
        print(f"Expanding dataset {num_responses_per_question}x...")
        expanded_dataset_list = []
        for i in range(num_responses_per_question):
            expanded_dataset_list.append(dataset)
        
        expanded_dataset = concatenate_datasets(expanded_dataset_list)
        
        # 3. Generate Responses
        print(f"Generating responses using {response_model}...")
        generator_kwargs = {
            "model_name": response_model,
            "generation_params": {"max_tokens": 1024, "temperature": 0.8},
        }
        if config.BACKEND:
            generator_kwargs["backend"] = config.BACKEND
        if config.BACKEND_PARAMS:
            generator_kwargs["backend_params"] = config.BACKEND_PARAMS

        response_generator = PersonalizedResponseGenerator(**generator_kwargs)
        dataset_with_responses = response_generator(expanded_dataset)
        
        # 4. Judge Responses
        print(f"Judging responses using {judge_model}...")
        judge_kwargs = {
            "model_name": judge_model,
        }
        if config.BACKEND:
            judge_kwargs["backend"] = config.BACKEND
        if config.BACKEND_PARAMS:
            judge_kwargs["backend_params"] = config.BACKEND_PARAMS

        judge_generator = PerceivabilityTestGenerator(**judge_kwargs)
        dataset_with_judgments = judge_generator(dataset_with_responses.dataset)
        
        # 5. Form DPO Pairs
        print("Forming DPO pairs...")
        df = dataset_with_judgments.dataset.to_pandas()
        grouped = df.groupby("question")
        
        dim_pairs_count = 0
        for question, group in grouped:
            winners = group[group["reward"] == 1]
            losers = group[group["reward"] == 0]
            
            if len(winners) > 0 and len(losers) > 0:
                for _, winner in winners.iterrows():
                    for _, loser in losers.iterrows():
                        all_dpo_pairs.append({
                            "prompt": winner["question"],
                            "chosen": winner["personalized_response"],
                            "rejected": loser["personalized_response"],
                            "dimension": winner["dimension_name"],
                            "persona": winner["ground_truth_persona"]
                        })
                        dim_pairs_count += 1
        print(f"Generated {dim_pairs_count} pairs for {dimension_name}")

    # 6. Save Combined Dataset
    print(f"\nTotal DPO pairs generated: {len(all_dpo_pairs)}")
    
    if len(all_dpo_pairs) > 0:
        import pandas as pd
        from datasets import Dataset as HFDataset
        
        dpo_df = pd.DataFrame(all_dpo_pairs)
        dpo_dataset = HFDataset.from_pandas(dpo_df)
        
        safe_response_model = response_model.replace("/", "-")
        # Use "All-Dimensions" or similar in name to indicate it's combined
        output_name = f"{config.HF_USERNAME}/PersonaSignal-DPO-Pairs-All-{safe_response_model}"
        print(f"Pushing combined DPO dataset to: {output_name}")
        dpo_dataset.push_to_hub(output_name)
    else:
        print("No pairs formed across all dimensions!")

if __name__ == "__main__":
    generate_all_dpo_pairs()
