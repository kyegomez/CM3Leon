# CM3Leon: Autoregressive Multi-Modal Model for Text and Image Generation

This repository hosts the open-source implementation of CM3Leon, a state-of-the-art autoregressive multi-modal model for text and image generation. The model is introduced in the paper "Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning".

## Overview

CM3Leon is a transformer-based autoregressive model designed for multi-modal tasks, specifically text and image generation. The model is trained in two stages, using a large diverse multimodal dataset and augmented retrieval pretraining. It also implements contrastive decoding to enhance the quality of the generated samples.

Key Features of CM3Leon:

- Retrieval augmented pretraining on a large diverse multimodal dataset.
- Two-stage training: pretraining and supervised fine-tuning.
- Contrastive decoding for enhanced sample quality.

CM3Leon sets a new benchmark in text-to-image generation, outperforming comparable models while requiring 5x less computational resources.

## Getting Started

The following sections provide a detailed analysis of the model architecture, the necessary resources, and the steps needed to replicate the CM3Leon model.

### Requirements

Replicating CM3Leon involves several critical components and requires proficiency in the following areas:

- Large-scale distributed training of transformer models using a significant number of GPUs/TPUs.
- Efficient data loading and preprocessing to handle extensive multimodal datasets.
- Memory optimization techniques to accommodate large models within the GPU memory.
- Custom tokenizer implementation for both text and image modalities.
- Setting up a retrieval infrastructure for dense retrieval during pretraining.
- Developing a fine-tuning framework to handle mixed text-image tasks.
- Inference optimizations such as compiler-accelerated decoders, lower precision computing, and batching.

### System Architecture

The CM3Leon implementation comprises:

- A distributed training framework, preferably TensorFlow or PyTorch.
- High-performance compute infrastructure (HPC cluster with GPUs/TPUs).
- A retrieval index and dense retriever module for augmentation.
- Data pipelines for efficient preprocessing and loading.
- Custom code for tokenizers and the CM3 model architecture.
- Fine-tuning framework and relevant task datasets.
- Serving infrastructure for low-latency inference.

Implementing these components involves challenges such as efficient utilization of large compute clusters, minimizing data loading and preprocessing bottlenecks, optimizing memory usage during training and inference, and ensuring low latency serving.

### Model Architecture

The architecture of CM3Leon includes:

- Text and Image Tokenizers: Custom text tokenizer trained on CommonCrawl data and Image tokenizer that encodes 256x256 images into 1024 tokens.
- Special Tokens: Usage of `<break>` token to indicate modality transitions.
- Retrieval Augmentation: Using a bi-encoder based on CLIP to retrieve relevant text and images from the memory bank.
- Autoregressive Decoder-only Transformer: Standard transformer architecture similar to GPT models.
- Two-Stage Training: Pretraining with retrieval augmentation and supervised finetuning on text-image tasks via instruction tuning.
- Contrastive Decoding: Modified contrastive decoding for better sample quality.

The model size ranges from 350M to 7B parameters.

### Data 

For successful implementation, CM3Leon requires:

- A large (100M+ examples) diverse multimodal dataset like Shutterstock for pretraining.
- A mixture of text and image tasks with accompanying datasets for finetuning.
- Efficient and scalable data loading that does not bottleneck model training.
- Preprocessing steps like resizing images to 256x256 pixels and text tokenization.

### Training

CM3Leon's training process involves:

- Pretraining with retrieval augmentation and CM3 objective.
- Supervised finetuning on text-image tasks.
- Efficient distributed training infrastructure for large-scale model training.
- Hyperparameter tuning for learning rates, batch sizes, optimizers, etc.

### Inference

For efficient inference, consider:

- Using compiler-accelerated decoders like FasterTransformer.
- Other optimizations like lower precision (FP16/INT8) and batching.
- Efficient implementation of contrastive decoding.

## Contributing

This repository welcomes contributions. Feel free to submit pull requests, create issues, or suggest any enhancements.

## Support

If you encounter any issues or need further clarification, please create an issue in the GitHub issue tracker.

## License

CM3Leon is open-sourced under the [MIT license](LICENSE).

