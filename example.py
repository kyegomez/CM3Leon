import torch
from cm3.model import CM3

img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))

model = CM3()
output = model(text, img)


