import math 
import torch 
from torch import einsum, _nnpack_available
import torch.nn.functional as F
from torch import nn
from einops import rearrange
import copy
from pathlib import PurePath
from tqdm import tqdm_gui
from beartype import beartype
from beartype.typing import Tuple, Optional

from einops import rearrange, repeat, reduce, unpack
from einops.layers.torch import Rearrange, Reduce


#helpers
def exists(val):
    return val is not None


#decorators
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

def defaults(val, d):
    return val if exists(val) else d

#tensor helpers

def log(t, eps=1e-20):
    return torch.log(t.clamp(min = eps))

def masked_mean(seq, mask=None, dim=1, keepdim=True):
    if not exists(mask):
        return seq.mean(dim=dim)
    
    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean


#sampling helpers

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature = 1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.einsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float("-inf")
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


class LoRA(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        r=8,
        alpha=None
    ):
        super().__init__()
        alpha = defaults(alpha, r)
        self.scale = alpha / r

        self.A = nn.Parameter(torch.randn(dim, r))
        self.B = nn.Parameter(torch.zeros(r, dim_out))



#reward model
@beartype

class RewardModel(nn.Module):
    def __init__(
            self,
            model: Andromeda,
            dropout=0.1,
            num_binned_output = 0.,
            use_lora = True,
            lora_r = 8,
            reward_lora_scope = 'reward',
    ):
        super().__init__()
        
        self.model = copy.deepcopy(Andromeda)
        self.model.set_dropout(dropout)

        self.reward_lora_scope = reward_lora_scope is use_lora else None

        if exists(self.reward_lora_scope):
            self.model.add_finetune_params(reward_lora_scope, lora_r = lora_r)

        dim = model.dim

        self.binned_output = num_binned_output > 1

        self.prompt_embed = nn.Parameter(torch.zeros(1, 1, dim))
        self.response_embed = nn.Parameter(torch.zeros(1, 1, dim))


        if self.binned_output:
            self.to_pred = nn.Linear(dim, num_binned_output)
        else:
            self.to_pred = nn.Sequential(
                nn.Linear(dim, 1, bias=False),
                Rearrange('... 1 -> ...')
            )

    def load(self, path):
        path = Path(path)
        assert path.exists()
        self.load_state_dict(torch.load(str(path)))

    def finetune_parameters(self):
        return (
            *self.to_pred.parameters(),
            *(self.model.finetune_parameters(self.reward_lora_scope) if exists(self.reward_lora_scope) else model.parameters())
        )
    
    
    def forward(
            self,
            x,
            mask=None,
            prompt_mask=None,
            prompt_lengths=None,
            labels=None,
            sample=False,
            sample_temperature=1.,
            disable_lora=False
    ):
        assert not (exists(prompt_mask) and exists(prompt_lengths))

        #derive prompt mask from prompt lengths

        if exists(prompt_lengths):
            batch, seq_len = x.shape
            arange = torch.arange(seq_len, device = x.device)
            prompt_mask = repeat(arange, 'n -> n n', b = batch) > rearrange(prompt_lengths, 'b -> b 1')

        #rward model should have an understand of which section is prompt and which section is repsonse

        extra_embed = None

        if exists(prompt_mask):
            extra_embed = torch.where(
                rearrange(prompt_mask, 'b n -> b n 1'),
                self.prompt_embed,
                self.response_embed
            )

        embeds = self.model(
            x,
        )