import math
import multiprocessing
import os
import collections

from datetime import timedelta
from functools import partial
from itertools import chain

import torch

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from datasets import concatenate_datasets, load_dataset

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)

from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import (AutoTokenizer, default_data_collator,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup, set_seed)

from datasets import Dataset

# from stable_adamw import StableAdamWUnfused
# sd

from optimus_prime import TransformerWrapper, Decoder, AutoregressiveWrapper
from optimus_prime import AndromedaEmbedding

from lion_pytorch import Lion
from sophia import SophiaG
import numpy as np

# constants

class CFG:
    BATCH_SIZE: int = 3 # 3
    GRADIENT_ACCUMULATE_EVERY: int = 1
    SEED: int = 42
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 1e-2
    SEQ_LEN: int = 8192 # 8192
    NUM_CPU: int = multiprocessing.cpu_count()
    USE_PRETOKENIZED: bool = True
    USE_ACTIVATION_CHECKPOINTING: bool = True
    RESUME_FROM_CHECKPOINT: str = None
    CHECKPOINTING_STEPS: int = 1000
    OUTPUT_DIR: str = "output"
    ENTITY_NAME: str = "nicolo" # Put your wandb username here

# helpers

def print_num_params(model, accelerator: Accelerator):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Number of parameters in model: {n_params}")

def fsdp_activation_checkpointing(
    model, accelerator: Accelerator, offload_to_cpu=False
):

    accelerator.print(f"Using FSDP activation checkpointing")

    # check_fn = lambda submodule: isinstance(submodule, ParallelTransformerBlock)

    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        offload_to_cpu=offload_to_cpu,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper)


def get_lr_scheduler_with_warmup(
    optimizer, scheduler_type, num_warmup_steps, max_train_steps, grad_accumulate_every
):
    NUM_WARMUP_STEPS = num_warmup_steps
    GRADIENT_ACCUMULATE_EVERY = grad_accumulate_every

    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
            num_training_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
            num_training_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY
        )
    else:
        raise ValueError(
            "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}".format(
                scheduler_type
            )
        )


def build_dataloaders():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    content_column = 'text'

    dataset = load_dataset("sentiment140", split="train")
    dataset = dataset.remove_columns([col for col in dataset.column_names if col != content_column])

    tokenized_dataset = dataset.map(
        lambda example: tokenizer([t + tokenizer.eos_token for t in example[content_column]]),
        batched=True,
        num_proc=CFG.NUM_CPU,
        remove_columns=[content_column]
    )

    block_size = CFG.SEQ_LEN

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {}

        for k in examples.keys():
            concatenated_examples[k] = list(chain(*examples[k]))

        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        return result

    train_dataset = tokenized_dataset.map(
        group_texts, batched=True, num_proc=CFG.NUM_CPU
    )

    return train_dataset

# main

def TrainAndromeda():
    # accelerator

    timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))

    accelerator = Accelerator(
        gradient_accumulation_steps=CFG.GRADIENT_ACCUMULATE_EVERY,
        mixed_precision="fp16", # Switch to bf16
        log_with="wandb",
        kwargs_handlers=[timeout]
    )

    accelerator.init_trackers(
        project_name="andromeda",
        config={
            "batch_size": CFG.BATCH_SIZE,
            "gradient_accumulate_every": CFG.GRADIENT_ACCUMULATE_EVERY,
            "learning_rate": CFG.LEARNING_RATE,
            "seq_len": CFG.SEQ_LEN,
        },
        init_kwargs={"wandb": {"entity": CFG.ENTITY_NAME}}
    )

    accelerator.print(f"Total GPUS: {accelerator.num_processes}")

    # set seed

    set_seed(CFG.SEED)

    # Create the tokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # instantiate andromeda

    model = TransformerWrapper(
        num_tokens=64007,
        max_seq_len=8192,
        use_abs_pos_emb=False,
        tokenizer=tokenizer, # !
        embedding_provider=AndromedaEmbedding(),
        attn_layers = Decoder(
            dim=128, # 2048
            depth=8, # 16
            dim_head=128,
            heads=8,
            alibi_pos_bias=True,
            alibi_num_heads=4,
            rotary_xpos=True,
            attn_flash = True,
            # deepnorm=True,
            shift_tokens=1,
            attn_one_kv_head = True,
            qk_norm=True,
            attn_qk_norm=True,
            attn_qk_norm_dim_scale=True # set this to True, in addition to `attn_qk_norm = True`
        )
    ).to(accelerator.device)

    model = AutoregressiveWrapper(model).to(accelerator.device)

    #optim = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    optim = SophiaG(model.parameters(), lr=1e-5, weight_decay=1e-1)

    print_num_params(model, accelerator)

    if CFG.USE_ACTIVATION_CHECKPOINTING:
        fsdp_activation_checkpointing(model, accelerator)

    # dataloaders

    if CFG.USE_PRETOKENIZED:
        d0 = load_dataset("conceptofmind/c4_0-to-20_neox_with_eos_8k", split="train")
        d1 = load_dataset("conceptofmind/c4_21-to-40_neox_with_eos_8k", split="train")
        d2 = load_dataset("conceptofmind/c4_41-to-60_neox_with_eos_8k", split="train")
        d3 = load_dataset("conceptofmind/c4_61-to-80_neox_with_eos_8k", split="train")
        d4 = load_dataset("conceptofmind/c4_81-to-100_neox_with_eos_8k", split="train")

        train_dataset = concatenate_datasets([d0, d1, d2, d3, d4])
    else:
        train_dataset = build_dataloaders()

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE, collate_fn=default_data_collator,
    )

    # optimizer

    # optim = decoupled_optimizer(
    #     model,
    #     learning_rate=CFG.LEARNING_RATE,
    #     weight_decay=CFG.WEIGHT_DECAY,
    #     beta_1=0.9,
    #     beta_2=0.95,
    #     use_adamw=False,
    # )

    # Determine number of training steps

    max_train_steps = math.ceil(len(train_loader) / CFG.GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps: {max_train_steps}")

    # lr scheduler
    # We cant decide on an actual number

    NUM_WARMUP_STEPS = int(max_train_steps * 0.01)
    accelerator.print(f"Num warmup steps: {NUM_WARMUP_STEPS}")

    lr_scheduler = get_lr_scheduler_with_warmup(
        optimizer=optim,
        scheduler_type="cosine",
        num_warmup_steps=NUM_WARMUP_STEPS,
        max_train_steps=max_train_steps,
        grad_accumulate_every=CFG.GRADIENT_ACCUMULATE_EVERY
    )

    # prepare

    model, optim, train_loader, lr_scheduler = accelerator.prepare(
        model, optim, train_loader, lr_scheduler
    )

    # checkpoint scheduler

    accelerator.register_for_checkpointing(lr_scheduler)

    # I do not know why Huggingface recommends recalculation of max_train_steps

    max_train_steps = math.ceil(len(train_loader) / CFG.GRADIENT_ACCUMULATE_EVERY)
    accelerator.print(f"Max train steps recalculated: {max_train_steps}")

    # Total batch size for logging

    total_batch_size = (
        CFG.BATCH_SIZE * accelerator.num_processes * CFG.GRADIENT_ACCUMULATE_EVERY
    )
    accelerator.print(f"Total batch size: {total_batch_size}")

    # resume training

    progress_bar = tqdm(
        range(max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    if CFG.RESUME_FROM_CHECKPOINT:
        if CFG.RESUME_FROM_CHECKPOINT is not None or CFG.RESUME_FROM_CHECKPOINT != "":
            accelerator.print(f"Resuming from checkpoint {CFG.RESUME_FROM_CHECKPOINT}")
            accelerator.load_state(CFG.RESUME_FROM_CHECKPOINT)
            path = os.path.basename(CFG.RESUME_FROM_CHECKPOINT)
        
        training_difference = os.path.splitext(path)[0]

        # need to multiply `gradient_accumulation_steps` to reflect real steps
        resume_step = (
            int(training_difference.replace("step_", ""))
            * CFG.GRADIENT_ACCUMULATE_EVERY
        )

    if CFG.RESUME_FROM_CHECKPOINT and resume_step is not None:
        train_loader = accelerator.skip_first_batches(train_loader, resume_step)
        completed_steps += resume_step
        progress_bar.update(resume_step)

    # training

    model.train()

    for step, batch in enumerate(train_loader):
        with accelerator.accumulate(model):
            inputs = batch["input_ids"].to(accelerator.device)
            _, loss = model(inputs, return_loss=True)
            accelerator.backward(loss)

            # print(loss.item())

            accelerator.log({"loss": loss.item()}, step=step)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 0.5)

            optim.step()
            lr_scheduler.step()
            optim.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            completed_steps += 1

        if isinstance(CFG.CHECKPOINTING_STEPS, int):
            if completed_steps % CFG.CHECKPOINTING_STEPS == 0:
                output_dir = f"step_{completed_steps }"
                if CFG.OUTPUT_DIR is not None:
                    output_dir = os.path.join(CFG.OUTPUT_DIR, output_dir)
                accelerator.save_state(output_dir)

        if completed_steps >= max_train_steps:
            break

    # end training

    accelerator.print(f"Training Finished")
    accelerator.end_training()

    # save final model

    # accelerator.print(f"Saving model to {CFG.OUTPUT_DIR}")
    if CFG.OUTPUT_DIR is not None:
        base_path = f'{CFG.OUTPUT_DIR}/final'

        if not os.path.exists(base_path):
            os.makedirs(base_path)

        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        with accelerator.main_process_first():
            accelerator.save(
                unwrapped_model.state_dict(), os.path.join(base_path, 'final_model.pt')
            )

if __name__ == "__main__":
    TrainAndromeda()