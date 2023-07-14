# CM3Leon: Scaling Autoregressive Multi-Modal Models via Pretraining and Instruction Tuning

This is an open source implementation of CM3Leon, an autoregressive multi-modal model for text and image generation presented in the paper "Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning".

## Model Architecture

CM3Leon uses a transformer-based architecture with the following key components:

- **Text and image tokenizers**: For text, a custom tokenizer is trained on CommonCrawl data. For images, the tokenizer from [Gafni et al. 2022](https://arxiv.org/abs/2203.13131) which encodes 256x256 images into 1024 tokens.

- **Retrieval augmentation**: A bi-encoder dense retriever based on CLIP is used to retrieve relevant text and images from a memory bank. Retrieval promotes diversity through query dropout and filtering highly similar documents. 

- **Autoregressive decoder-only transformer**: The transformer is initialized similarly to GPT models. During pretraining, the [CM3 objective](https://arxiv.org/abs/2201.07520) is used for infilling and generation.

- **Two-stage training**: 
  - **Pretraining**: Models are pretrained on a large proprietary Shutterstock dataset using retrieval augmentation and the CM3 objective.
  - **Supervised fine-tuning**: Additional multi-modal datasets are used to fine-tune the model on various text-image tasks via instruction tuning.

- **Contrastive decoding**: A modified version of contrastive decoding provides improved sample quality by blending conditional and unconditional generations.

The models are trained at 350M, 760M and 7B parameter sizes.

## Replicating Key Results

To replicate the key results:

- Obtain a large multi-modal dataset for pretraining like Shutterstock. Alternatively, use CC3M or Conceptual Captions.

- Implement the model architecture above. Models should be pretrained with retrieval augmentation.

- Fine-tune the pretrained models on multi-modal supervised data via instruction tuning. Use the datasets described in the paper.

- Evaluate on image generation (FID on COCO, retrieval augmentation), and text generation (captioning, VQA)

- Compare different decoding strategies like contrastive decoding vs classifier-free guidance.

- Run ablations on model size, retrieval augmentation, and decoding methods.

- Compare to baseline models like DALL-E, Stable Diffusion, etc. on scale, efficiency, and accuracy.

The most important factors are pretraining with a large and diverse multi-modal dataset, and finetuning on strong supervision via instruction tuning. Contrastive decoding also gives quality improvements.
