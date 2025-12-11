import sys
import os
from pathlib import Path
import random
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from dotenv import load_dotenv
import pandas as pd

# Add parent directory to path to import config and inference modules
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from inference.collect_response import PersonalizedResponseGenerator
from inference.test_perceivability import PerceivabilityTestGenerator
from inference.leakage_check import LeakageChecker

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
            input_dataset_name = f"{config.HF_USERNAME}/PersonaSignal-PersonaQuestions-{formatted_dim}-DPO"
            
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
        
        # 5. Check Leakage
        print(f"Checking for leakage using {judge_model}...")
        leakage_checker = LeakageChecker(**judge_kwargs)
        
        # We only need to check rows where reward == 1
        # But to keep it simple and consistent with the leakage_check script logic,
        # we can filter or just run on all (but that's expensive).
        # Let's filter like the script does.
        
        df = dataset_with_judgments.dataset.to_pandas()
        # Convert to list of dicts for curator
        rows_to_check = df[df["reward"] == 1].to_dict(orient="records")
        
        if rows_to_check:
            print(f"Checking {len(rows_to_check)} rows where reward == 1...")
            # Run leakage check
            try:
                checked_rows = leakage_checker(rows_to_check).dataset.to_list()
            except Exception as e:
                print(f"Error running leakage checker: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: assume no leakage if check fails, or skip?
                # For debugging, let's re-raise to see if we can get more info
                raise e
            
            # Create a map for quick lookup
            # Assuming uniqueness of question + personalized_response + persona might be tricky if duplicates exist.
            # But we are processing list to list.
            # Let's reconstruct the dataframe with final_reward.
            
            # Better approach: Iterate and update.
            # Or just use the logic from leakage_check.py to combine.
            
            checked_map = {
                (r["question"], r["personalized_response"], r["ground_truth_persona"]): r 
                for r in checked_rows
            }
            
            final_rows = []
            for i, row in df.iterrows():
                key = (row["question"], row["personalized_response"], row["ground_truth_persona"])
                if row["reward"] == 1 and key in checked_map:
                    final_rows.append(checked_map[key])
                else:
                    # Not checked (reward != 1), so final_reward = reward (0)
                    row_dict = row.to_dict()
                    row_dict["leakage_detected"] = False
                    row_dict["leakage_rationale"] = "Skipped check (reward != 1)"
                    row_dict["final_reward"] = row["reward"]
                    final_rows.append(row_dict)
            
            df = pd.DataFrame(final_rows)
        else:
            print("No rows with reward=1 to check.")
            df["final_reward"] = df["reward"]

        # 6. Form DPO Pairs
        print("Forming DPO pairs...")
        grouped = df.groupby("question")
        
        dim_pairs_count = 0
        for question, group in grouped:
            # Winners: Must have final_reward == 1 (Correct persona AND No leakage)
            winners = group[group["final_reward"] == 1]
            
            # Losers: final_reward == 0 (Either Wrong persona OR Leakage)
            losers = group[group["final_reward"] == 0]
            
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

    # 7. Save Combined Dataset
    print(f"\nTotal DPO pairs generated: {len(all_dpo_pairs)}")
    
    if len(all_dpo_pairs) > 0:
        dpo_df = pd.DataFrame(all_dpo_pairs)
        dpo_dataset = HFDataset.from_pandas(dpo_df)
        
        safe_response_model = response_model.replace("/", "-")
        output_name = f"{config.HF_USERNAME}/PersonaSignal-DPO-Pairs-All-{safe_response_model}"
        print(f"Pushing combined DPO dataset to: {output_name}")
        dpo_dataset.push_to_hub(output_name)
    else:
        print("No pairs formed across all dimensions!")

if __name__ == "__main__":
    generate_all_dpo_pairs()
