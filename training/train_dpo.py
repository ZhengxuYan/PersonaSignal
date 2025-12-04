"""
Direct Preference Optimization (DPO) training for PersonaSignal using Tinker
"""

import asyncio
import logging
import os
import time
from typing import Any, cast, List

import sys
from pathlib import Path

import chz
import tinker
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from tinker_cookbook import checkpoint_utils, cli_utils, model_info
from tinker_cookbook.eval.evaluators import Evaluator, EvaluatorBuilder
from tinker_cookbook.supervised.train import run_evals
from tinker_cookbook.supervised.types import ChatDatasetBuilder, SupervisedDataset, ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.format_colorized import format_colorized
from tinker_cookbook.utils.lr_scheduling import compute_schedule_lr_multiplier
from tinker_cookbook.utils.misc_utils import timed

logger = logging.getLogger(__name__)

from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset
from tinker_cookbook.renderers import get_renderer, Message
from tinker_cookbook.supervised.common import datum_from_tokens_weights

# Add parent directory to path to import config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

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

def PersonaSignalDpoDataset(
    dataset_name: str, 
    tokenizer: Tokenizer, 
    max_length: int, 
    renderer_name: str,
    batch_size: int
) -> SupervisedDataset:
    dataset = load_dataset(dataset_name, split="train")
    renderer = get_renderer(renderer_name, tokenizer)
    
    def example_to_data(example: dict[str, Any]) -> list[tinker.Datum]:
        prompt = example["prompt"]
        chosen = example["chosen"]
        rejected = example["rejected"]
        persona = example["persona"]
        
        system_prompt = PERSONALIZED_RESPONSE_SYSTEMPROMPT_TEMPLATE.format(persona=persona)
        
        chosen_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]
        rejected_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected}
        ]
        
        chosen_tokens, chosen_weights = renderer.build_supervised_example(chosen_messages)
        rejected_tokens, rejected_weights = renderer.build_supervised_example(rejected_messages)
        
        return [
            datum_from_tokens_weights(chosen_tokens, chosen_weights, max_length),
            datum_from_tokens_weights(rejected_tokens, rejected_weights, max_length)
        ]

    return SupervisedDatasetFromHFDataset(
        dataset,
        batch_size=batch_size,
        flatmap_fn=example_to_data
    )

@chz.chz
class PersonaSignalDpoDatasetBuilder(ChatDatasetBuilder):
    dataset_name: str
    
    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        # Use tokenizer from common_config if available, or default
        tokenizer = self.tokenizer # This uses common_config.model_name_for_tokenizer
            
        dataset = PersonaSignalDpoDataset(
            self.dataset_name, 
            tokenizer, 
            max_length=self.common_config.max_length or 1024,
            renderer_name=self.common_config.renderer_name,
            batch_size=self.common_config.batch_size
        )
        return dataset, None

@chz.chz
class Config:
    """Configuration for Direct Preference Optimization (DPO) training."""

    # Required parameters
    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Dataset parameters
    max_length: int = 1024
    batch_size: int = 2
    
    dataset_builder: ChatDatasetBuilder = chz.field(
        default_factory=lambda: PersonaSignalDpoDatasetBuilder(
            dataset_name=f"{config.HF_USERNAME}/PersonaSignal-DPO-Pairs-All-{config.RESPONSE_GEN_MODEL.replace('/', '-')}",
            common_config=ChatDatasetBuilderCommonConfig(
                model_name_for_tokenizer= "meta-llama/Llama-3.1-8B-Instruct",
                renderer_name=model_info.get_recommended_renderer_name("meta-llama/Llama-3.1-8B-Instruct"),
                max_length=1024,
                batch_size=2,
            )
        )
    )
    load_checkpoint_path: str | None = None

    # Training parameters
    learning_rate: float = 1e-5
    lr_schedule: str = "linear"
    num_epochs: int = 1
    dpo_beta: float = 0.1

    # Model parameters
    lora_rank: int = 32

    # Infrastructure parameters
    num_replicas: int = 1
    base_url: str | None = None

    # Checkpointing and evaluation
    evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    infrequent_evaluator_builders: list[EvaluatorBuilder] = chz.field(default_factory=list)
    save_every: int = 20
    eval_every: int = 10
    infrequent_eval_every: int = 100

    # Adam optimizer parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Logging parameters
    wandb_project: str | None = "PersonaSignal-DPO"
    wandb_name: str | None = "dpo-run"


# ... [Include the rest of the functions from the example: create_dpo_clients, compute_dpo_loss, do_update, main, print_example] ...
# For brevity, I will copy them below.

def create_dpo_clients(
    config: Config,
    resume_info: dict[str, Any] | None = None,
) -> tuple[tinker.TrainingClient, tinker.SamplingClient]:
    service_client = tinker.ServiceClient(base_url=config.base_url)
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )

    if resume_info:
        training_client.load_state_with_optimizer(resume_info["state_path"]).result()
        logger.info(f"Resumed DPO training from {resume_info['state_path']}")
    elif config.load_checkpoint_path:
        training_client.load_state(config.load_checkpoint_path).result()
        logger.info(f"Loaded weights from {config.load_checkpoint_path}")
    
    reference_client = training_client.save_weights_and_get_sampling_client("reference")
    return training_client, reference_client


def compute_dpo_loss(
    chosen_logprobs: list[torch.Tensor],
    rejected_logprobs: list[torch.Tensor],
    chosen_ref_logprobs: list[torch.Tensor],
    rejected_ref_logprobs: list[torch.Tensor],
    dpo_beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    chosen_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(chosen_logprobs, chosen_ref_logprobs, strict=True)]
    )
    rejected_log_ratio = torch.stack(
        [lp - rlp for lp, rlp in zip(rejected_logprobs, rejected_ref_logprobs, strict=True)]
    )

    losses = -torch.log(torch.sigmoid(dpo_beta * (chosen_log_ratio - rejected_log_ratio)))
    loss = losses.mean()

    accuracy = (chosen_log_ratio > rejected_log_ratio).float().mean().item()
    chosen_rewards = dpo_beta * chosen_log_ratio
    rejected_rewards = dpo_beta * rejected_log_ratio
    margin = dpo_beta * (chosen_rewards - rejected_rewards).mean().item()

    metrics = {
        "dpo_loss": loss.item(),
        "accuracy": accuracy,
        "margin": margin,
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }

    return loss, metrics


def do_update(
    epoch_idx: int,
    batch_idx: int,
    n_batches: int,
    total_steps: int,
    config: Config,
    training_client: tinker.TrainingClient,
    reference_client: tinker.SamplingClient,
    evaluators: list[Evaluator],
    infrequent_evaluators: list[Evaluator],
    dataset: SupervisedDataset,
    ml_logger: ml_log.Logger,
    log_path: str,
    tokenizer: Tokenizer,
):
    start_time = time.time()
    step = epoch_idx * n_batches + batch_idx
    metrics: dict[str, int | float | str] = {"epoch": epoch_idx}

    if step % config.save_every == 0 and step > 0:
        with timed("save_checkpoint", metrics):
            save_result = checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=log_path,
                kind="both",
                loop_state={"epoch": epoch_idx, "batch": batch_idx},
            )
        if "state_path" in save_result:
            metrics["state_path"] = save_result["state_path"]

    learning_rate = config.learning_rate * compute_schedule_lr_multiplier(
        lr_schedule=config.lr_schedule, step=step, total_steps=total_steps
    )
    adam_params = tinker.AdamParams(
        learning_rate=learning_rate,
        beta1=config.adam_beta1,
        beta2=config.adam_beta2,
        eps=config.adam_eps,
    )

    if config.eval_every > 0 and step % config.eval_every == 0:
        with timed("evals", metrics):
            eval_metrics = asyncio.run(run_evals(evaluators, training_client, step))
        metrics.update(eval_metrics)

    if config.infrequent_eval_every > 0 and step % config.infrequent_eval_every == 0:
        with timed("infrequent_evals", metrics):
            eval_metrics = asyncio.run(run_evals(infrequent_evaluators, training_client, step))
        metrics.update(eval_metrics)

    with timed("get_batch", metrics):
        data = dataset.get_batch(batch_idx)

    chosen_data = [datum for i, datum in enumerate(data) if i % 2 == 0]
    rejected_data = [datum for i, datum in enumerate(data) if i % 2 == 1]

    if step == 0:
        for i in range(min(10, len(chosen_data))):
            print_example(chosen_data[i], tokenizer, "Chosen")
            print_example(rejected_data[i], tokenizer, "Rejected")

    with timed("get_ref_logprobs", metrics):
        full_sequences = []
        for datum in data:
            target_tokens = datum.loss_fn_inputs["target_tokens"].data
            if target_tokens:
                full_sequence = datum.model_input.append_int(int(target_tokens[-1]))
                full_sequences.append(full_sequence)
            else:
                full_sequences.append(datum.model_input)

        async def compute_all_ref_logprobs():
            return await asyncio.gather(
                *[reference_client.compute_logprobs_async(seq) for seq in full_sequences]
            )

        all_ref_logprobs = asyncio.run(compute_all_ref_logprobs())
        all_ref_logprob_seqs = [torch.tensor(logprobs[1:]) for logprobs in all_ref_logprobs]
        chosen_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(0, len(data), 2)]
        rejected_ref_logprob_seqs = [all_ref_logprob_seqs[i] for i in range(1, len(data), 2)]

    def dpo_loss_fn(
        data: list[tinker.Datum], logprobs_list: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        chosen_logprob_seqs = [logprobs_list[i] for i in range(0, len(data), 2)]
        rejected_logprob_seqs = [logprobs_list[i] for i in range(1, len(data), 2)]

        chosen_logprobs = []
        chosen_ref_logprobs = []
        rejected_logprobs = []
        rejected_ref_logprobs = []

        for i in range(len(chosen_data)):
            chosen_logprob_seq = chosen_logprob_seqs[i]
            chosen_ref_logprob_seq = chosen_ref_logprob_seqs[i]
            chosen_weights = torch.tensor(chosen_data[i].loss_fn_inputs["weights"].data)
            chosen_logprob = torch.dot(chosen_logprob_seq.float(), chosen_weights.float())
            chosen_ref_logprob = torch.dot(chosen_ref_logprob_seq.float(), chosen_weights.float())
            chosen_logprobs.append(chosen_logprob)
            chosen_ref_logprobs.append(chosen_ref_logprob)

            rejected_logprob_seq = rejected_logprob_seqs[i]
            rejected_ref_logprob_seq = rejected_ref_logprob_seqs[i]
            rejected_weights = torch.tensor(rejected_data[i].loss_fn_inputs["weights"].data)
            rejected_logprob = torch.dot(rejected_logprob_seq.float(), rejected_weights.float())
            rejected_ref_logprob = torch.dot(
                rejected_ref_logprob_seq.float(), rejected_weights.float()
            )
            rejected_logprobs.append(rejected_logprob)
            rejected_ref_logprobs.append(rejected_ref_logprob)

        return compute_dpo_loss(
            chosen_logprobs=chosen_logprobs,
            rejected_logprobs=rejected_logprobs,
            chosen_ref_logprobs=chosen_ref_logprobs,
            rejected_ref_logprobs=rejected_ref_logprobs,
            dpo_beta=config.dpo_beta,
        )

    with timed("step", metrics):
        backward_result = training_client.forward_backward_custom(data, dpo_loss_fn).result()
        dpo_metrics = backward_result.metrics
        training_client.optim_step(adam_params).result()

    metrics.update(
        num_pairs=len(chosen_data),
        num_tokens=sum(datum.model_input.length for datum in data),
        learning_rate=learning_rate,
        progress=step / total_steps,
        **dpo_metrics,
    )

    metrics["time/total"] = time.time() - start_time
    ml_logger.log_metrics(metrics=metrics, step=step)


def main(config: Config):
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_epoch = resume_info["epoch"]
        start_batch = resume_info["batch"]
    else:
        start_epoch = 0
        start_batch = 0

    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )
    training_client, reference_client = create_dpo_clients(config, resume_info)
    tokenizer = get_tokenizer(config.model_name)

    dataset, maybe_test_dataset = config.dataset_builder()
    n_batches = len(dataset)
    total_steps = n_batches * config.num_epochs

    evaluators = [evaluator() for evaluator in config.evaluator_builders]
    infrequent_evaluators = [evaluator() for evaluator in config.infrequent_evaluator_builders]
    logger.info(
        f"Training for {n_batches} batches x {config.num_epochs} epochs = {n_batches * config.num_epochs} steps"
    )

    for epoch_idx in range(start_epoch, config.num_epochs):
        logger.info(msg=f"Starting epoch {epoch_idx}")
        dataset.set_epoch(seed=epoch_idx)

        for batch_idx in range(start_batch if epoch_idx == start_epoch else 0, n_batches):
            do_update(
                epoch_idx=epoch_idx,
                batch_idx=batch_idx,
                n_batches=n_batches,
                total_steps=total_steps,
                config=config,
                training_client=training_client,
                reference_client=reference_client,
                evaluators=evaluators,
                infrequent_evaluators=infrequent_evaluators,
                dataset=dataset,
                ml_logger=ml_logger,
                log_path=config.log_path,
                tokenizer=tokenizer,
            )

    if start_epoch < config.num_epochs:
        checkpoint_utils.save_checkpoint(
            training_client=training_client,
            name="final",
            log_path=config.log_path,
            kind="both",
            loop_state={"epoch": config.num_epochs, "batch": n_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    ml_logger.close()
    logger.info("DPO training completed successfully")


def print_example(datum: tinker.Datum, tokenizer: Tokenizer, label: str = ""):
    int_tokens = list(datum.model_input.to_ints())
    weights = datum.loss_fn_inputs["weights"].data
    logger.info(f"\n{label} Example:")
    logger.info(format_colorized(int_tokens, cast(list[float], weights), tokenizer))

if __name__ == "__main__":
    load_dotenv()
    chz.nested_entrypoint(main)
