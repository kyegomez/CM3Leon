import math
import multiprocessing
import os
from datetime import timedelta
from functools import partial
from itertools import chain

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from accelerate import Accelerator
from accelerate.utils import (DummyOptim, DummyScheduler,
                              InitProcessGroupKwargs)
from datasets import concatenate_datasets, load_dataset
from lion_pytorch import Lion
# from palm_rlhf_pytorch import PaLM
from torch.nn import LayerNorm
# from palm_rlhf_pytorch.palm import LayerNorm, TransformerWrapper

from torch.nn import LayerNorm
from Andromeda.optimus_prime import TransformerWrapper, AutoregressiveWrapper, AndromedaEmbedding, Decoder

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy
)


from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoTokenizer, default_data_collator,
                          get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup, set_seed)

# from palm.stable_adamw import StableAdamWUnfused
from Andromeda.utils.stable_adamw import StableAdamWUnfused

from Andromeda.optimus_prime import TransformerWrapper, AutoregressiveWrapper, AndromedaEmbedding, Decoder


class TrainAndromeda:
    class CFG:
        BATCH_SIZE = 3
        GRADIENT_ACCUMULATE_EVERY: int = 1
        SEED: int = 42
        LEARNING_RATE: float = 3e-4
        WEIGHT_DECAY: float = 0.1
        SEQ_LEN: int = 8192
        NUM_CPU: int = multiprocessing.cpu_count()
        USE_DEEPSPEED: bool = True
        USE_FSDP: bool = True
        USE_PRETOKENIZED: bool = True
        USE_ACTIVATION_CHECKPOINTING: bool = True
        RESUME_FROM_CHECKPOINT: str = True
        CHECKPOINTING_STEPS: int = 1000
        OUTPUT_DIR: str = "YOUR_OUTPUT_DIR"
        ENTITY_NAME: str = "YOUR_ENTITY_NAME"

    @staticmethod
    def print_num_params(model, accelerator: Accelerator):
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        accelerator.print(f"Number of parameters in model: {n_params}")

    @staticmethod
    def activation_checkpointing(
        model: torch.nn.Module,
        offload_to_cpu: bool = False,
        accelerator: Accelerator = None,
    ):
        if accelerator is not None:
            accelerator.print(f"Using activation checkpointing")
        check_fn = lambda submodule: isinstance(submodule, TransformerWrapper)
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=offload_to_cpu,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )

    @staticmethod
    def fsdp(
        model: torch.nn.Module,
        auto_wrap: bool = False,
        mp: str = "fp32",
        shard_strat: str = "NO_SHARD",
    ):
        if auto_wrap:
            andromeda_auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    TransformerWrapper,
                },
            )
        else:
            andromeda_auto_wrap_policy = None

        if mp == "bf16":
            mp_fsdp = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        elif mp == "fp16":
            mp_fsdp = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
        elif mp == "fp32":
            mp_fsdp = MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'bf16', 'fp16' or 'fp32', got: {}".format(
                    mp
                )
            )

        if shard_strat == "SHARD_GRAD":
            sharding_strat_fsdp = ShardingStrategy.SHARD_GRAD_OP 
        elif shard_strat == "FULL_SHARD":
            sharding_strat_fsdp = ShardingStrategy.FULL_SHARD
        elif shard_strat == "NO_SHARD":
            sharding_strat_fsdp = ShardingStrategy.NO_SHARD
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'SHARD_GRAD', 'FULL_SHARD' or 'NO_SHARD', got: {}".format(
                    shard_strat
                )
            )

        model = FullyShardedDataParallel(
            model,
            auto_wrap_policy=andromeda_auto_wrap_policy,
            mixed_precision=mp_fsdp,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            sharding_strategy=sharding_strat_fsdp,
            forward_prefetch=True,
            use_orig_params=True,
        )

        return model

    @staticmethod
    def get_lr_scheduler_with_warmup(
        optimizer: torch.optim.Optimizer,
        scheduler_type: str,
        num_warmup_steps: int,
        max_train_steps: int,
        grad_accumulate_every: int = 1,
        accelerator: Accelerator = None,
    ):
        NUM_WARMUP_STEPS = num_warmup_steps
        GRADIENT_ACCUMULATE_EVERY = grad_accumulate_every
        if accelerator is not None:
            accelerator.print(f"Using {scheduler_type} lr scheduler")
        if scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
                num_training_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY,
            )
        elif scheduler_type == "cosine":
            return get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=NUM_WARMUP_STEPS * GRADIENT_ACCUMULATE_EVERY,
                num_training_steps=max_train_steps * GRADIENT_ACCUMULATE_EVERY,
            )
        else:
            raise ValueError(
                "Invalid scheduler_type. Expected 'linear' or 'cosine', got: {}".format(
                    scheduler_type
                )
            )

    @staticmethod
    def decoupled_optimizer(
        model: torch.nn.Module,
        learning_rate: float,
        weight_decay: float,
        beta_1: float,
        beta_2: float,
        optimizer_type: str,
        use_fsdp: bool = True,
        accelerator: Accelerator = None,
    ):
        """
        Decouples the optimizer from the training process.

        This function sets up the optimizer for the model by creating two groups of parameters:
        one for weight decay and one without weight decay. Then, it initializes the optimizer
        with these two groups of parameters.

        Args:
            model (Module): The model whose parameters are optimized.
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the optimizer.
            beta_1 (float): The exponential decay rate for the 1st moment estimates.
            beta_2 (float): The exponential decay rate for the 2nd moment estimates.
            optimizer_type (str): The type of the optimizer. Can be 'lion', 'adamw', or 'stable_adamw'.
            use_fsdp (bool, optional): If True, the optimizer will work with fully sharded data parallelism. Defaults to True.
            accelerator (Accelerator, optional): The accelerator from HuggingFace's Accelerate library. Defaults to None.

        Returns:
            Optimizer: The initialized optimizer.

        Raises:
            ValueError: If the optimizer type is not 'lion', 'adamw' or 'stable_adamw'.
        """
        accelerator.print(f"Using {optimizer_type} optimizer")
        # Create an empty dictionary called param_dict to store the model's named parameters.
        param_dict = {}
        # Iterate over the model's named parameters and populate the param_dict with key-value pairs.
        for param_name, param in model.named_parameters():
            param_dict[param_name] = param

        # Separate the model's named modules into two groups: decay and no_decay.

        # Create an empty list to store the names of the LayerNorm and Embedding layer weights with no weight decay.
        no_decay = []

        if use_fsdp:
            exclude_module = "_fsdp_wrapped_module.token_emb"
        else:
            exclude_module = "token_emb"

        # Iterate through the named modules of the model.
        for module_name, module in model.named_modules():
            # Check if the current module is an instance of any of the desired types (LayerNorm or torch.nn.Embedding).
            for ndim in [LayerNorm, torch.nn.Embedding]:
                if isinstance(module, ndim):
                    # If torch.nn.Embedding, append its name with a ".weight" suffix to the no_decay list.
                    if module_name == exclude_module:
                        no_decay.append(f"{module_name}.weight")
                    else:
                        # If the module is an instance of LayerNorm
                        no_decay.append(f"{module_name}.gamma")
                    # Exit the inner loop since the desired module has been found.
                    break

        # Create an empty list to store the names of the Linear layer weights with weight decay.
        decay = []

        # Iterate through the named modules of the model.
        for module_name, module in model.named_modules():
            # Check if the current module is an instance of the desired type (torch.nn.Linear).
            for ndim in [torch.nn.Linear]:
                if isinstance(module, ndim):
                    # If the module is an instance of torch.nn.Linear, append its name with a ".weight" suffix to the decay list.
                    decay.append(f"{module_name}.weight")
                    # Exit the inner loop since the desired module has been found.
                    break

        # Create two separate lists of model parameters: decay_param and no_decay_param.
        # The decay_param list contains the parameters that should have weight decay applied.
        # The no_decay_param list contains the parameters that should not have weight decay applied, excluding the 'to_logits.weight' parameter.

        # Create an empty list called decay_param to store the parameters with weight decay.
        decay_param = []

        if use_fsdp:
            exclude_param = "_fsdp_wrapped_module.to_logits.weight"
        else:
            exclude_param = "to_logits.weight"

        # Iterate over the decay list, which contains the names of the parameters with weight decay.
        for param in decay:
            # Check if the current parameter is not 'to_logits.weight'.
            # Append the corresponding parameter from param_dict to the decay_param list.

            if param != exclude_param:
                decay_param.append(param_dict[param])

        # Create an empty list called no_decay_param to store the parameters without weight decay.
        no_decay_param = []

        # Iterate over the no_decay list, which contains the names of the parameters without weight decay.
        for param in no_decay:
            # Append the corresponding parameter from param_dict to the no_decay_param list.
            no_decay_param.append(param_dict[param])

        # Create a list called grouped_params that contains two dictionaries.
        # The first dictionary has the decay_param list and the corresponding weight_decay value.
        # The second dictionary has the no_decay_param list and a weight_decay value of 0.0.
        grouped_params = [
            {"params": decay_param, "weight_decay": weight_decay},
            {"params": no_decay_param, "weight_decay": 0.0},
        ]

        # Create a variable called optimizer that stores an instance of the optimizer.
        if optimizer_type == "lion":
            optimizer = Lion(grouped_params, lr=learning_rate, betas=(beta_1, beta_2),)
        elif optimizer_type == "adamw":
            optimizer = AdamW(grouped_params, lr=learning_rate, betas=(beta_1, beta_2),)
        elif optimizer_type == "deepspeed":
            optimizer = DummyOptim(grouped_params, lr=learning_rate, betas=(beta_1, beta_2),)
        elif optimizer_type == "stable_adamw":
            optimizer = StableAdamWUnfused(
                grouped_params, lr=learning_rate, betas=(beta_1, beta_2),
            )
        else:
            raise ValueError(
                "Invalid optimizer_type. Expected 'lion', 'adamw', 'deepspeed' or 'stable_adamw', got: {}".format(
                    optimizer_type
                )
            )

        # Return the optimizer.
        return optimizer


    @staticmethod
    def build_dataloaders():
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        dataset = load_dataset("openwebtext", split="train")

        tokenized_dataset = dataset.map(
            lambda example: tokenizer([t + tokenizer.eos_token for t in example["text"]]),
            batched=True,
            num_proc=TrainAndromeda.CFG.NUM_CPU,
            remove_columns=["text"],
        )

        block_size = TrainAndromeda.CFG.SEQ_LEN

        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result

        train_dataset = tokenized_dataset.map(
            group_texts, batched=True, num_proc=TrainAndromeda.CFG.NUM_CPU,
        )

        return train_dataset

    @staticmethod
    def build_pre_tokenized():
        d0 = load_dataset("conceptofmind/c4_0-to-20_neox_with_eos_8k", split="train")
        return d0

    @staticmethod
    def Train():
        timeout = InitProcessGroupKwargs(timeout=timedelta(seconds=1_000_000))
        accelerator = Accelerator(
            gradient_accumulation_steps=TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY,
            mixed_precision="fp16",
            log_with="wandb",
            kwargs_handlers=[timeout],
        )
        accelerator.init_trackers(
            project_name="Andromeda",
            config={
                "batch_size": TrainAndromeda.CFG.BATCH_SIZE,
                "gradient_accumulate_every": TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY,
                "learning_rate": TrainAndromeda.CFG.LEARNING_RATE,
                "seq_len": TrainAndromeda.CFG.SEQ_LEN,
            },
            init_kwargs={"wandb": {"entity": TrainAndromeda.CFG.ENTITY_NAME}},
        )
        accelerator.print(f"Total GPUS: {accelerator.num_processes}")
        set_seed(TrainAndromeda.CFG.SEED)
        model = TransformerWrapper(
            num_tokens=64007,
            max_seq_len=8192,
            use_abs_pos_emb=False,
            embedding_provider=AndromedaEmbedding(),
            attn_layers = Decoder(
                dim=2560,
                depth=32,
                dim_head=128,
                heads=24,
                alibi_pos_bias=True,
                alibi_num_heads=12,
                rotary_xpos=True,
                attn_flash=True,
                deepnorm=True,
                shift_tokens=1,
                attn_one_kv_head=True,
                qk_norm=True,
                attn_qk_norm=True,
                attn_qk_norm_dim_scale=True
            )
        ).to(accelerator.device)
        model = AutoregressiveWrapper(model).to(accelerator.device)
        TrainAndromeda.print_num_params(model, accelerator)

        if TrainAndromeda.CFG.USE_FSDP:
            model = TrainAndromeda.fsdp(
                model,
                mp="fp16",
                shard_strat="SHARD_GRAD"
            )

        if TrainAndromeda.CFG.USE_ACTIVATION_CHECKPOINTING:
            TrainAndromeda.activation_checkpointing(model, accelerator)

        model = accelerator.prepare(model)

        if TrainAndromeda.CFG.USE_PRETOKENIZED:
            train_dataset = TrainAndromeda.build_pre_tokenized()
        else:
            train_dataset = TrainAndromeda.build_dataloaders()

        train_loader = DataLoader(
            train_dataset, batch_size=TrainAndromeda.CFG.BATCH_SIZE, collate_fn=default_data_collator,
        )

        optim = TrainAndromeda.decoupled_optimizer(
            model=model,
            learning_rate=TrainAndromeda.CFG.LEARNING_RATE, 
            weight_decay=TrainAndromeda.CFG.WEIGHT_DECAY, 
            beta_1=0.90, 
            beta_2=0.95, 
            optimizer_type='deepspeed',  
            use_fsdp=True,
            accelerator=accelerator
        )

        max_train_steps = math.ceil(len(train_loader) / TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY)
        accelerator.print(f"Max train steps: {max_train_steps}")

        NUM_WARMUP_STEPS = int(max_train_steps * 0.01)
        accelerator.print(f"Num warmup steps: {NUM_WARMUP_STEPS}")

        if TrainAndromeda.CFG.USE_DEEPSPEED:
            lr_scheduler = DummyScheduler(
                optim, 
                total_num_steps=max_train_steps * accelerator.num_processes, 
                warmup_num_steps=NUM_WARMUP_STEPS
            )
        else:
            lr_scheduler = TrainAndromeda.get_lr_scheduler_with_warmup(
                optimizer=optim,
                scheduler_type="cosine",
                num_warmup_steps=NUM_WARMUP_STEPS,
                max_train_steps=max_train_steps,
                grad_accumulate_every=TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY,
            )

        optim, train_loader, lr_scheduler = accelerator.prepare(
            optim, train_loader, lr_scheduler
        )

        accelerator.register_for_checkpointing(lr_scheduler)

        max_train_steps = math.ceil(len(train_loader) / TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY)
        accelerator.print(f"Max train steps recalculated: {max_train_steps}")

        total_batch_size = (
            TrainAndromeda.CFG.BATCH_SIZE * accelerator.num_processes * TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY
        )
        accelerator.print(f"Total batch size: {total_batch_size}")

        progress_bar = tqdm(
            range(max_train_steps), disable=not accelerator.is_local_main_process
        )
        completed_steps = 0

        if TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT:
            if TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT is not None or TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT != "":
                accelerator.print(f"Resuming from checkpoint {TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT}")
                accelerator.load_state(TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT)
                path = os.path.basename(TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT)
            training_difference = os.path.splitext(path)[0]

            resume_step = (
                int(training_difference.replace("step_", ""))
                * TrainAndromeda.CFG.GRADIENT_ACCUMULATE_EVERY
            )

        if TrainAndromeda.CFG.RESUME_FROM_CHECKPOINT and resume_step is not None:
            train_loader = accelerator.skip_first_batches(train_loader, resume_step)
            completed_steps += resume_step
            progress_bar.update(resume_step)

        model.train()
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                inputs = batch["input_ids"].to(accelerator.device)
                loss = model(inputs, return_loss=True)
                accelerator.backward(loss)

                accelerator.log({"loss": loss.item()}, step=step)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optim.step()
                lr_scheduler.step()
                optim.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(TrainAndromeda.CFG.CHECKPOINTING_STEPS, int):
                if completed_steps % TrainAndromeda.CFG.CHECKPOINTING_STEPS == 0:
                    output_dir = f"step_{completed_steps }"
                    if TrainAndromeda.CFG.OUTPUT_DIR is not None:
                        output_dir = os.path.join(TrainAndromeda.CFG.OUTPUT_DIR, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= max_train_steps:
                break

        accelerator.end_training()

        if TrainAndromeda.CFG.OUTPUT_DIR is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            with accelerator.main_process_first():
                accelerator.save(
                    unwrapped_model.state_dict(), f"{TrainAndromeda.CFG.OUTPUT_DIR}/final/final_model.pt"
                )

def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '9994'
    
    # [CRITICAL] Pay attention to this when scaling to multiple GPUs and clusters
    
    os.environ['RANK']       = str(0) # Number of nodes (servers)
    os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())

    torch.distributed.init_process_group()
    
    Train()

if __name__ == "__main__":
    main()