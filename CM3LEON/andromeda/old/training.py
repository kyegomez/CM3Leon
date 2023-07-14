#quantization + paralleism
import time

import torch
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from accelerate import Accelerator

from rich.progress import Progress


from lion_pytorch import Lion
# from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
from optimus_prim import TransformerWrapper, Decoder, AutoregressiveWrapper

from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    CPUOffload,
)

from torch.distributed.fsdp.wrap import (
    default_auto_wrap_policy,
)

from transformers import AutoTokenizer

#logging
import boto3


#training
import wandb

from torch.utils.tensorboard import SummaryWriter

class CustomGPTNeoXTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

custom_tokenizer = CustomGPTNeoXTokenizer()

Andromeda = TransformerWrapper(
    num_tokens=64007,
    max_seq_len=8192,
    use_abs_pos_emb = False,
    tokenizer=custom_tokenizer,
    attn_layers = Decoder(
        dim=2048,
        depth=6,
        heads=16,
        alibi_pos_bias=True,
        alibi_num_heads=8,
        rotary_xpos=True,
        attn_flash = True,
        deepnorm=True,
        shift_tokens=1,
        attn_one_kv_head = True,
        qk_norm=True
    )
)

Andromeda = AutoregressiveWrapper(Andromeda)



AWS_ACCESS_KEY_ID=""
AWS_SECRET_ACCESS_KEY="d"


def save_model_to_s3(model, bucket_name, key_prefix, step):
    s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    model_path = f"checkpoint_at_step_{step}.pt"
    torch.save(model.state_dict(), model_path)
    s3.upload_file(model_path, bucket_name, f"{key_prefix}/{model_path}")



def count_number_of_parameters(model, only_trainable: bool = True) -> int:
    if only_trainable:
        num_params: int = sum(p.numel()
                              for p in model.parameters() if p.requires_grad)
    else:
        num_params: int = sum(p.numel() for p in model.parameters() if p)
    return int(num_params)



def prep_sample(sample):
    title = sample["title"]
    text = sample["text"]
    return {
        "title": title,
        "text": text
    }


def train(args):

    if args.use_ddp:
        dist.init_process_group(backend="nccl")


    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    #v1
    model = Andromeda()
    if args.use_ddp:
        model = DistributedDataParallel(model)
    else:
        model = DataParallel(model)

    fsdp_model = FullyShardedDataParallel(
        model(),
        fsdp_auto_wrap_policy=default_auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=True),
    )

    fsdp_model = fsdp_model.to(accelerator.device)

    #device count
    if torch.cuda.device_count() > 1:
        print(f"Let's use ${torch.cuda.device_count()} GPUS")




    optimizer = Lion(model.parameters(), lr=args.learning_rate / 3, weight_decay=args.weight_decay * 3)
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # tokenizer = KosmosTokenizer()

    #====================> load data #====================> load data #====================> load data 


    dataset = load_dataset("the_pile_books3")

    # dataset = dataset.map(prep_sample, num_proc=8)
    dataset = dataset.map(prep_sample, num_proc=8)


    #new removed columns
    remove_columns = ['title']


    dataset = dataset.map(Andromeda.decoder.tokenizer, batched=True,
                          batch_size=128, remove_columns=remove_columns)

    train_dataloader = DataLoader(
        dataset, collate_fn=default_data_collator, batch_size=args.batch_size, pin_memory=True
    )



    #====================> load data #====================> load data #====================> load data #====================> load data 

    fsdp_model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(fsdp_model, train_dataloader, optimizer,
                                                                           lr_scheduler)
    fsdp_model.train()
    accelerator.register_for_checkpointing(lr_scheduler)

    accelerator.print(
        f"Number of parameters: {count_number_of_parameters(model):,}")
    accelerator.print(
        f"Number of trainable parameters: {count_number_of_parameters(model, only_trainable=True):,}")

    # Log model and optimizer parameters to wandb
    accelerator.init_trackers(project_name="Andromeda")

    #wandb
    wandb.init(project="Andromeda", config=args)
    
    #init tensorboard writer
    tb_writer = SummaryWriter()


    train_loader = iter(train_dataloader)
    epoch_loss = 0
    total_loss = 0
    start_time = time.time()

    with Progress() as progress:
        task = progress.add_task("[red]Training...", total=args.max_steps)
        for step in range(0, args.max_steps):
            batch_start = time.time()
            batch = next(train_loader)
            outputs = fsdp_model(**batch, self_attn_padding_mask=batch["attention_mask"])
            # Shift so that tokens < n predict n
            outputs = torch.cat([outputs[:, :1], outputs[:, 67:]], dim=1).contiguous()
            # shift_logits = outputs[..., :-1, :].contiguous()
            # shift_labels = batch["labels"][..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            one_hot_labels = torch.nn.functional.one_hot(batch["labels"][:, 1:], num_classes=32002).float()
            loss = loss_fct(outputs[:,:-1], one_hot_labels)

            epoch_loss += loss.detach().float()

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            batch_end = time.time()
            logs = {
                "loss": loss.item(),
                "perplexity": torch.exp(loss).item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "examples": args.batch_size * (step + 1),
                "examples_per_second": args.batch_size / (batch_end - batch_start),
            }
            if step % args.log_every == args.log_every - 1:
                #log metrics to wandb
                wandb.log(logs, step=step)

                #log metrics to tensorboard 
                                # Log metrics to TensorBoard
                tb_writer.add_scalar("loss", logs["loss"], step)
                tb_writer.add_scalar("perplexity", logs["perplexity"], step)
                tb_writer.add_scalar("lr", logs["lr"], step)
                tb_writer.add_scalar("examples", logs["examples"], step)
                tb_writer.add_scalar("examples_per_second", logs["examples_per_second"], step)

                #accelerator
                accelerator.log(logs, step=step)
                progress.update(task, advance=1, description=f"Step Loss: {loss.item():.5f} "
                                                             f"| Mean Loss: {(total_loss + epoch_loss) / step:.5f} "
                                                             f"| Mean PPL: {torch.exp((total_loss + epoch_loss) / step):.2f} "
                                                             f"| Examples: {args.batch_size * (step + 1)} "
                                                             f"| Examples/s: {args.batch_size / (batch_end - batch_start):.2f} "
                                                             f"| Elapsed: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

            if step % args.save_every == args.save_every - 1:
                train_epoch_loss = epoch_loss / args.save_every
                total_loss += epoch_loss
                epoch_loss = 0

                accelerator.log({
                    "train_ppl": torch.exp(train_epoch_loss),
                    "train_epoch_loss": train_epoch_loss,
                }, step=step)

                progress.print(f"Saving checkpoint at step {step}...")
                accelerator.save_state(
                    f"{args.checkpoint_dir}/checkpoint_at_step_{step}/")
                
                #save the model weights to s3 
                save_model_to_s3(model, "kosmostraining", "kosmosv1/checkpoints", step)
                print(f"Saved to s3: {save_model_to_s3} ")

        #finish tensorboard writer
        tb_writer.close()

        #finish wnabd run
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_ddp", action="store_true", help="Use DistributedDataParallel")

    args = parser.parse_args()

    train(args)