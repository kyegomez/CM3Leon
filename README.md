[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)


# CM3Leon: Autoregressive Multi-Modal Model for Text and Image Generation (wip)

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/CM3Leon)](https://github.com/kyegomez/CM3Leon/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/CM3Leon)](https://github.com/kyegomez/CM3Leon/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/CM3Leon)](https://github.com/kyegomez/CM3Leon/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/CM3Leon)](https://github.com/kyegomez/CM3Leon/blob/master/LICENSE)
[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/CM3Leon)](https://twitter.com/intent/tweet?text=Excited%20to%20introduce%20CM3Leon,%20the%20all-new%20Multi-Modal%20model%20with%20the%20potential%20to%20revolutionize%20automation.%20Join%20us%20on%20this%20journey%20towards%20a%20smarter%20future.%20%23CM3Leon%20%23Multi-Modal&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FCM3Leon)
[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FCM3Leon)
[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FCM3Leon&title=Introducing%20CM3Leon%2C%20the%20All-New%20Multi-Modal%20Model&summary=CM3Leon%20is%20the%20next-generation%20Multi-Modal%20model%20that%20promises%20to%20transform%20industries%20with%20its%20intelligence%20and%20efficiency.%20Join%20us%20to%20be%20a%20part%20of%20this%20revolutionary%20journey%20%23CM3Leon%20%23Multi-Modal&source=)
![Discord](https://img.shields.io/discord/999382051935506503)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FCM3Leon&title=Exciting%20Times%20Ahead%20with%20CM3Leon%2C%20the%20All-New%20Multi-Modal%20Model%20%23CM3Leon%20%23Multi-Modal) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FCM3Leon&t=Exciting%20Times%20Ahead%20with%20CM3Leon%2C%20the%20All-New%20Multi-Modal%20Model%20%23CM3Leon%20%23Multi-Modal)
[![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FCM3Leon&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=CM3Leon%2C%20the%20Revolutionary%20Multi-Modal%20Model%20that%20will%20Change%20the%20Way%20We%20Work%20%23CM3Leon%20%23Multi-Modal)
[![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=I%20just%20discovered%20CM3Leon,%20the%20all-new%20Multi-Modal%20model%20that%20promises%20to%20revolutionize%20automation.%20Join%20me%20on%20this%20exciting%20journey%20towards%20a%20smarter%20future.%20%23CM3Leon%20%23Multi-Modal%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FCM3Leon)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kyegomez/CM3Leon/blob/main/google_colab.ipynb)


CM3Leon is a transformer-based autoregressive model designed for multi-modal tasks, specifically text and image generation. The model is trained in two stages, using a large diverse multimodal dataset and augmented retrieval pretraining. It also implements contrastive decoding to enhance the quality of the generated samples.

[CM3LEON, PAPER LINK](https://ai.meta.com/research/publications/scaling-autoregressive-multi-modal-models-pretraining-and-instruction-tuning/)

* Please Help with this open source implementation in the Agora discord, ![Discord](https://img.shields.io/discord/999382051935506503)
* This implementation is still not finished.

## Install

```pip3 install cm3```

---

## Usage & Example

To start with CM3Leon in a PyTorch environment:

```python
import torch
from cm3.model import CM3

# usage
img = torch.randn(1, 3, 256, 256)
caption = torch.randint(0, 20000, (1, 1024))

model = CM3()

output = model(img, caption)
print(output.shape)  # (1, 1024, 20000)


```

This repository hosts the open-source implementation of CM3Leon, a state-of-the-art autoregressive multi-modal model for text and image generation. The model is introduced in the paper "Scaling Autoregressive Multi-Modal Models: Pretraining and Instruction Tuning".

---

## Overview

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


Here is a markdown table with the datasets used in the paper along with additional metadata and source links:

| Dataset | Domain | Size | Source | 
|-|-|-|-|  
| Shutterstock | Images and captions | 3 billion text tokens, licensed image data | Proprietary dataset, described in paper |
| MS-COCO | Image captioning | 591K image-caption pairs | [Microsoft COCO Captions](https://cocodataset.org/#captions-2015) |
| Flickr30k | Image captioning | 144K image-caption pairs | [Flickr30k Entities](https://www.robots.ox.ac.uk/~vgg/data/flickr30k/) |  
| Image Paragraph | Dense image captioning | 14K images with paragraph captions | [Image Paragraph dataset](https://cs.stanford.edu/people/ranjaykrishna/imcap/) |
| Localized Narratives | Image paragraph captioning | 164K images with localized narratives | [Localized Narratives](https://github.com/jponttuset/localizing-narratives) |
| VQA2 | Visual question answering | 1.3M images with question-answer pairs | [VQA2 dataset](https://visualqa.org/download.html) |  
| VizWiz | Visual question answering for blind users | 92K images with question-answer pairs | [VizWiz dataset](https://vizwiz.org/) |
| OKVQA | Knowledge-based VQA | 26K images with question-answer pairs | [OK-VQA dataset](https://okvqa.allenai.org/) |
| ScienceQA | Scientific visual QA | 6K images with multi-choice QA pairs | [ScienceQA](https://allenai.org/data/science-qa) |


The model was trained and evaluated on several datasets including MS-COCO [...] (Chen et al., 2015), Flickr30k [...] (Young et al., 2014), etc.

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


## HyperParameters
```Model size # L dmodel Seq Length Batch LR Warmup Steps # GPUs # Tokens
350M 24 1024 4096 8M 6e-04 1500 256 1.4T
760M 24 1536 4096 8M 5e-04 1500 256 1.9T
7B 32 4096 4096 8M 1.2e-04 1500 512 2.4T
```

## SuperVised FineTuning parameters
``` 
Model # GPUS Seq Length Batch Size LR Warm-up Steps # Tokens
CM3Leon-760m 64 4096 2M 5e-05 150 30B
CM3Leon-7b 128 4096 2M 5e-05 150 30B
```

# Innovations in the paper:

* Conditional text + image generation with objective function + contrastive top k decoding

* Multi-Modality models need to be dynamic they can't just generate the types of data they were trained on they need to be able to adapt to user needs therefore multi-modality models should be conditional, if prompted the model will generate text and or images, this is the future.

## Contributing

This repository welcomes contributions. Feel free to submit pull requests, create issues, or suggest any enhancements.

## Support

If you encounter any issues or need further clarification, please create an issue in the GitHub issue tracker.

## License

CM3Leon is open-sourced under the [MIT license](LICENSE).

# Roadmap

* Implement Objective function where multi-modal inputs are transformed into an infilling instance by masking specific spans and relocating them to the end. 

* Implement a next token prediction loss, -log p(x input) 

* Implement TopP sampling 

* Implement Free Guidance CFG => directing an unconditional sample towards a conditional sample. Replace text with mask token from cm3 objective for uncoditional sampling so that during inference 2 concurrent tokens tsreams are generated a conditional stream, which is contigent on the input text and an unconditional token stream which is conditioned on a mask token Where

```python
Logits, cond = T(ty | ty), logit.uncond = T(ty | <mask>)
logits.cf = logits.uncond + a.c * (logits.cond - logits.uncond)

T = transformer
ty = output tokens
tx = conditional input text <mask>
<mask> = no input text + replacement with a mask token
a.c = scaling factor
```

* Implement Contrastive Decoding TopK => 
```
V(t.y < .i) = {t.yi is in V: P.exp(t.yi | t.y<.i) >= a * kmax(p.exp(w|t.y<i))}
```

# Citation
