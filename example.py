import torch
from cm3.model import CM3

#usage
img = torch.randn(1, 3, 256, 256)
caption_tokens = torch.randint(0, 4)

model = CM3()
output = model(img, caption_tokens)
