import os
from logging import getLogger
from typing import List, Optional, Union

# from clipq import CLIPQ
from sentencepiece import SentencePieceProcessor

logger = getLogger()


class Tokenizer:
    """
    A SentencePieceTokenizer is a tokenizer that uses a pretrained SentencePiece model
    to convert text into tokens and vice versa.

    It includes the ability to add special tokens for infilling tasks and provides
    functionality to encode and decode text with or without implicit leading spaces.

    Parameters:
    - model_path (str): Path to the pretrained SentencePiece model file.

    Attributes:
    - n_words (int): Vocabulary size of the SentencePiece model.
    - bos_id (int): Token ID of the beginning-of-sentence (BOS) token.
    - eos_id (int): Token ID of the end-of-sentence (EOS) token.
    - pad_id (int): Token ID of the padding (PAD) token.
    - prefix_id (int, optional): Token ID of the prefix token. Default: None.
    - middle_id (int, optional): Token ID of the middle token. Default: None.
    - suffix_id (int, optional): Token ID of the suffix token. Default: None.
    - eot_id (int, optional): Token ID of the end-of-turn (EOT) token. Default: None.



    """

    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        # token IDs for special infilling tokens
        self.prefix_id: Optional[int] = self.sp_model.piece_to_id("▁<PRE>") or None
        self.middle_id: Optional[int] = self.sp_model.piece_to_id("▁<MID>") or None
        self.suffix_id: Optional[int] = self.sp_model.piece_to_id("▁<SUF>") or None
        self.eot_id: Optional[int] = self.sp_model.piece_to_id("▁<EOT>") or None

        # generates text until a modality break token is detected => then img is sampled
        self.break_id: Optional[int] = self.sp_model.piece_to_id("_<BREAK>") or None
        self.image_id: Optional[int] = self.sp_model.piece_to_id("_<IMG>") or None
        self.infill_id: Optional[int] = self.sp_model.piece_to_id("_<INFILL>") or None

        logger.info(
            f"BREAK ID: {self.break_id} - IMG ID: {self.image_id} - INFILL ID: {self.infill_id}"
        )

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} "
            f"- PRE ID: {self.prefix_id} - MID ID: {self.middle_id} - SUF ID: {self.suffix_id} - EOT ID: {self.eot_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def encode_infilling(self, s: str) -> List[int]:
        """Encode a string without an implicit leading space."""
        return self.sp_model.encode("☺" + s)[2:]

    def decode_infilling(self, t: List[int]) -> str:
        """Decode a string without an implicit leading space."""
        return self.sp_model.decode([self.sp_model.piece_to_id("☺")] + t)[1:]


# class CM3LeonTokenizer(Tokenizer):
#     """
#     CM3LeonTokenizer is a tokenizer that uses a pretrained SentencePiece model
#     to convert text into tokens and vice versa.

#     It includes the ability to add special tokens for infilling tasks and provides
#     functionality to encode and decode text with or without implicit leading spaces.

#     Parameters:
#     - model_path (str): Path to the pretrained SentencePiece model file.

#     Attributes:
#     - n_words (int): Vocabulary size of the SentencePiece model.
#     - bos_id (int): Token ID of the beginning-of-sentence (BOS) token.
#     - eos_id (int): Token ID of the end-of-sentence (EOS) token.
#     - pad_id (int): Token ID of the padding (PAD) token.
#     - prefix_id (int, optional): Token ID of the prefix token. Default: None.
#     - middle_id (int, optional): Token ID of the middle token. Default: None.
#     - suffix_id (int, optional): Token ID of the suffix token. Default: None.
#     - eot_id (int, optional): Token ID of the end-of-turn (EOT) token. Default: None.

#     Usage
#     -----
#     tokenizer = CM3LeonTokenizer(model_path="path/to/model")
#     tokens = tokenizer.encode("this is a description", "https://picsum.photos/800")
#     print(tokens)
#     """
#     def __init__(
#         self,
#         model_path: str
#     ):
#         super().__init__(model_path)
#         self.clip = CLIPQ(
#             model_path=model_path,
#             query_text="A photo of an image segment",
#         )

#     def encode(
#         self,
#         s: str = None,
#         image: str = None,
#     ) -> List[Union[int, float]]:
#         """Encodes the image and text into a sequence of tokens and embeddings"""
#         #encode text
#         text = self.encode(
#             s,
#             bos=True,
#             eos=False,
#         )

#         #get embeds for img
#         img = self.clip.run_from_url(
#             url=image,
#             h_splits=3,
#             v_splits=3
#         )

#         #combine text, tokens and image embeddings
#         #starting with a <break> token followed by img embeds
#         # and ending with a eos token

#         seq = text + [self.break_id] + img + [self.eos_id]
#         return seq
