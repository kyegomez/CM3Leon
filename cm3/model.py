import logging

import torch
import torch.nn as nn
from flamingo_pytorch import PerceiverResampler
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

from cm3.core.model import Andromeda

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if the modules are available
try:
    import bitsandbytes
except ImportError as e:
    logging.error(f"Failed to import module: {e}")
    raise




# Implement classes with type hints and error handling
class CM3LEONTokenizer:
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



class CM3LEON(nn.Module):
    """
    The main CM3LEON model class.

    Attributes:
        clip_model (CLIPModel): The CLIP model for image processing.
        embed (Embedding): The embedding layer for tokens.
        embed_positions: (PositionEmbedding): The positional embedding layer.
        
        output_projection (Linear): the output projection layer.
        config (DecoderConfig): The configuration for the decoder
        decoder (Decoder): The decoder module

        perceieve(PerceiverResampler): The PerceieverResampler module for image processing.
        image_proj (Linear): The image projection layer.
    """

    def __init__(self):
        super().__init__()

        # Instantiate Clip Vit-l/14
        try:
            self.clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-L-14-laion2B-s32B-b82K").vision_model
        except Exception as e:
            logging.error(f"Failed to initialize CLIP model: {e}")
            raise

        self.embed = bitsandbytes.nn.modules.Embedding(32002, 2048, padding_idx=1)

        self.output_projection = nn.Linear(2048, 32002, bias=False)
        nn.init.normal_(self.output_projection.weight, mean=0, std=2048**-0.5)
        
        try:
            self.decoder = Andromeda()
        except Exception as e:
            logging.error(f"Failed to initialize Decoder: {e}")
            raise

        self.perceive = PerceiverResampler(
            dim = 1024,
            depth = 2,
            dim_head = 64,
            heads = 8,
            num_latents = 64,
            num_media_embeds = 257
        )

        self.image_proj = nn.Linear(1024, 2048, bias=False)
        nn.init.normal_(self.image_proj.weight, mean=0, std=2048**-0.5)

    def forward(self, text_tokens: torch.Tensor, images: torch.Tensor, **kwargs):
        """
       The forward pass for the CM3LEON model.

       Args:
            text_tokens (torch.Tensor): The text tokens.
            images (torch.Tensor): The image tokens.

        Returns:
            The output of the decoder 
        
        """
        if not isinstance(text_tokens, torch.Tensor) or not isinstance(images, torch.Tensor):
            raise TypeError("text_tokens and images must be instances of torch.Tensor")

        try:
            images = self.clip_model(pixel_values=images)["last_hidden_state"]
            images = self.perceive(images).squeeze(1)
            images = self.image_proj(images)
        except Exception as e:
            logging.error(f"Failed during image processing: {e}")
            raise

        try:
            model_input = self.decoder(text_tokens)[0]
            model_input = torch.cat([model_input[:, 0:2], images, model_input[:, 2:]], dim=1)
        except Exception as e:
            logging.error(f"Failed during text processing: {e}")
            raise

        try:
            return self.decoder(model_input, padded_x=model_input[0])
        except Exception as e:
            logging.error(f"Failed during model forward pass: {e}")
            raise