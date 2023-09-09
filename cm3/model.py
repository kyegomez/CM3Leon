import logging

import torch
from torch import nn
from torch.nn import Module
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import AutoTokenizer, CLIPProcessor


from zeta.nn.architecture.transformer import AutoregressiveWrapper, Decoder, Encoder, Transformer, ViTransformerWrapper


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')




# Implement classes with type hints and error handling
class CM3Tokenizer:
    """
    A tokenizer class for the CM3LEON model

    Attributes:
        processor(CLIPProcessor): The processor to tokenize images
        tokenizer: (AutoTokenizer): The tokenizer to tokenize text 
        im_idx: (int): The Index of the "<image>" token.
        im_end_idx (int): The index of the "</image>" token.
        break_idx (int): The index of the "<break>" token.
    """

    def __init__(self):
        try:
            self.processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "EleutherAI/gpt-neox-20b",
                additional_special_tokens=["<image>", "</image>", "<break>"],
                eos_token="<eos>",
                pad_token="<pad>",
                extra_ids=0,
                model_max_length=8192
            )

            self.image_tokenizer = Compose([
                Resize((256, 256)),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        except Exception as e:
            logging.error(f"Failed to initialize AndromedaTokenizer: {e}")
            raise

        self.im_idx, self.im_end_idx, self.break_idx = self.tokenizer.convert_tokens_to_ids(["<image>", "</image>", "<break>"])

    def tokenize_texts(self, texts: str):
        """
        Tokenize given texts.

        Args:
            Texts (str): The Text to be tokenized

        
        Returns:
            A tuple containing the tokenized texts and only the text tokens.
        """
        try:
            texts =  self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
            # Add image tokens to text as "<s> <image> </image> <break> text </s>"
            special_tokens = torch.tensor([[self.im_idx, self.im_end_idx, self.break_idx]] * texts.shape[0])
            return torch.cat([texts[:, 0:1], special_tokens, texts[:, 1:]], dim=1), texts
        except Exception as e:
            logging.error(f"Failed to tokenize texts: {e}")
            raise

    def tokenize_images(self, images):
        """
        Tokenizes given images.

        Args:
            images: The images to be tokenized

        Returns:
            The tokenized images.
        
        """
        try:
            return self.processor(images=images, return_tensors="pt").pixel_values
        except Exception as e:
            logging.error(f"Failed to tokenize images: {e}")
            raise

    def tokenize(self, sample):
        """
        Tokenizes given sample.

        Args:
            Sample: The sample to be tokenized

        Returns:
            A dictionary containing the tokenized text tokens, images, labels, and attention mask.
        
        """
        try:
            text_tokens, only_text_tokens = self.tokenize_texts(sample["target_text"])
            attention_mask = text_tokens != self.tokenizer.pad_token_id
            dummy_image_features = torch.ones((text_tokens.shape[0], 64))
            attention_mask = torch.cat([dummy_image_features, attention_mask], dim=1)
            return {
                "text_tokens": text_tokens,
                "images": self.tokenize_images(sample["image"]),
                "labels": only_text_tokens,
                "attention_mask": attention_mask,
            }
        except Exception as e:
            logging.error(f"Failed to tokenize sample: {e}")
            raise


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
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads
            )
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
                cross_attend=True
            )
        )

        self.decoder = AutoregressiveWrapper(self.transformer)

    def mask_and_relocate(self, text_tokens):
        #mask image span
        text_tokens = text_tokens.masked_fill(text_tokens==self.im_idx, self.mask_token)

        #relocate to end
        image_span = text_tokens[text_tokens==self.im_end_idx].unsqueeze(1)
        text_tokens = torch.cat([text_tokens, image_span], dim=1)
        return text_tokens
    
    def cm3_loss(self, log_probs, labels):
        #cm3 loss prediction
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

