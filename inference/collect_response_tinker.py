import asyncio
import os
import sys
from pathlib import Path
import tinker
from tinker_cookbook import renderers, model_info, tokenizer_utils
from tinker_cookbook.completers import TinkerMessageCompleter
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
import pandas as pd

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add tinker-cookbook to path
sys.path.insert(0, "/Users/jasonyan/Desktop/PersonaSignal/.venv/src/tinker-cookbook")

import config


load_dotenv()

# Constants
SAMPLER_PATH = "tinker://13841bd2-2189-5a5d-b8d2-a43ff7830fb5:train:0/sampler_weights/final"
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

PERSONALIZED_RESPONSE_SYSTEMPROMPT_TEMPLATE = """
You are a helpful assistant responding to a user with the following characteristics: {persona}

**Critical Instructions:**
1. Adapt your response to naturally align with this user's background, preferences, and context
2. NEVER explicitly mention, reference, or describe the user's persona traits in your response
3. NEVER use phrases like "given your [trait]", "since you are [description]", or "as someone who [characteristic]"
4. Let the personalization emerge implicitly through:
   - The reasoning approach and depth you choose
   - The examples, priorities, or framing you emphasize
   - The assumptions you make about context
   - The level of detail and abstraction
   - The order and focus of your points

Your goal: Provide a response that would naturally suit this user without making the persona obvious or detectable through explicit mentions.
"""

async def generate_response(completer, question, persona):
    system_prompt = PERSONALIZED_RESPONSE_SYSTEMPROMPT_TEMPLATE.format(persona=persona)
    messages = [
        renderers.Message(role="system", content=system_prompt),
        renderers.Message(role="user", content=question),
    ]
    output = await completer(messages)
    return output["content"]

async def main():
    print(f"Generating personalized responses for dimension: {config.DIMENSION_NAME}")

    # Load dataset
    input_dataset_name = config.get_dataset_name_with_model(
        "questions", config.QUESTION_GEN_MODEL
    )
    print(f"Loading dataset from: {input_dataset_name}")
    dataset = load_dataset(input_dataset_name, split="train")

    print("Initializing Tinker Client...")
    service_client = tinker.ServiceClient(api_key=os.environ.get("TINKER_API_KEY"))
    
    # Get Tokenizer
    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL_NAME)

    # Create Sampling Client with DPO weights
    print(f"Creating sampling client for {SAMPLER_PATH}...")
    sampling_client = service_client.create_sampling_client(
        base_model=BASE_MODEL_NAME, 
        model_path=SAMPLER_PATH
    )
    
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    
    completer = TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=renderer,
        max_tokens=1024
    )

    print("Generating responses...")
    results = []
    
    # Process in batches or one by one. For simplicity with async, we can use gather.
    # But let's do chunks to avoid overwhelming the client if dataset is large.
    batch_size = 10
    
    # Convert dataset to list of dicts for easier handling
    data = [item for item in dataset]
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        print(f"Processing batch {i} to {min(i+batch_size, len(data))}...")
        
        tasks = []
        for item in batch:
            tasks.append(generate_response(completer, item["question"], item["ground_truth_persona"]))
        
        batch_responses = await asyncio.gather(*tasks)
        
        for item, response in zip(batch, batch_responses):
            result_item = item.copy()
            result_item["personalized_response"] = response
            results.append(result_item)

    # Create new dataset
    new_dataset = Dataset.from_list(results)
    
    # Push to HuggingFace Hub
    # We'll use a suffix to indicate it's the DPO model
    output_dataset_name = f"{config.HF_USERNAME}/PersonaSignal-PersonalizedResponse-{config.DIMENSION_NAME.replace('_', '-').title()}-DPO-Tinker"
    print(f"Pushing dataset to: {output_dataset_name}")
    new_dataset.push_to_hub(output_dataset_name)
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
