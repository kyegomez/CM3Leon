import logging

import torch
from torch import nn
from torch.nn import Module
from zeta.nn.architecture.auto_regressive_wrapper import AutoregressiveWrapper
from zeta.nn.architecture.transformer import (
    Decoder,
    Encoder,
    Transformer,
    ViTransformerWrapper,
)

# logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


# main model
class CM3(Module):
    """
    Andromeda is a transformer-based model architecture. It initializes with
    a Transformer and AutoregressiveWrapper with default or user-specified parameters.

    Initialize the model with specified or default parameters.
        Args:
        - num_tokens: Number of tokens in the vocabulary
        - max_seq_len: Maximum sequence length
        - dim: Dimension of the model
        - depth: Depth of the model
        - dim_head: Dimension of the model head
        - heads: Number of heads
        - use_abs_pos_emb: Whether to use absolute position embedding
        - alibi_pos_bias: Alibi position bias
        - alibi_num_heads: Number of alibi heads
        - rotary_xpos: Rotary position
        - attn_flash: Attention flash
        - deepnorm: Deep normalization
        - shift_tokens: Number of tokens to shift
        - attn_one_kv_head: Attention one key/value head
        - qk_norm: Query-key normalization
        - attn_qk_norm: Attention query-key normalization
        - attn_qk_norm_dim_scale: Attention query-key normalization dimension scale
    """

    def __init__(
        self,
        num_tokens=50432,
        max_seq_len=8192,
        dim=2560,
        depth=32,
        dim_head=128,
        heads=24,
        use_abs_pos_emb=False,
        alibi_pos_bias=True,
        alibi_num_heads=12,
        rotary_xpos=True,
        attn_flash=True,
        image_size=256,
        patch_size=32,
        attn_one_kv_head=True,  # multiquery attention
        qk_norm=True,
        attn_qk_norm=True,
        attn_qk_norm_dim_scale=True,
    ):
        super().__init__()

        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(dim=dim, depth=depth, dim_head=dim_head, heads=heads),
        )

        self.transformer = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                alibi_pos_bias=alibi_pos_bias,
                alibi_num_heads=alibi_num_heads,
                rotary_xpos=rotary_xpos,
                attn_flash=attn_flash,
                # attn_one_kv_head=attn_one_kv_head,
                # qk_norm=qk_norm,
                # attn_qk_norm=attn_qk_norm,
                # attn_qk_norm_dim_scale=attn_qk_norm_dim_scale,
                cross_attend=True,
            ),
        )

        self.decoder = AutoregressiveWrapper(self.transformer)

    def mask_and_relocate(self, text_tokens):
        # mask image span
        text_tokens = text_tokens.masked_fill(
            text_tokens == self.im_idx, self.mask_token
        )

        # relocate to end
        image_span = text_tokens[text_tokens == self.im_end_idx].unsqueeze(1)
        text_tokens = torch.cat([text_tokens, image_span], dim=1)
        return text_tokens

    def cm3_loss(self, log_probs, labels):
        # cm3 loss prediction
        loss = nn.NLLLoss()(log_probs, labels)
        return loss

    # def forward(self, text_tokens, img, **kwargs):
    #     try:
    #         encoded_img = self.encoder(img, return_embeddings=True)

    #         #mask and relocate image span in text tokens
    #         text_tokens = self.mask_and_relocate(text_tokens)

    #         #concat
    #         context = torch.cat([encoded_img, text_tokens], dim=1)

    #         #get log probs
    #         log_probs = self.decoder(context, **kwargs)

    #         #calculate cm3 loss
    #         loss = self.cm3_loss(log_probs, text_tokens)

    #         return loss
    #         # return self.decoder(text_tokens, context=encoded_img)
    #     except Exception as error:
    #         print(f"Failed in forward method: {error}")
    #         raise

    def forward(self, img, text):
        try:
            encoded = self.encoder(img, return_embeddings=True)
            return self.decoder(text, context=encoded)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise
