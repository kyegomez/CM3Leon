from PIL import Image
from torchvision import transforms

from pali import VitModel


def img_to_tensor(img: str = "pali.png", img_size: int = 256):
    # Load image
    image = Image.open(img)

    # Define a transforms to convert the image to a tensor and apply preprocessing
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.Resize((img_size, img_size)),  # Resize the image to 256x256
            transforms.ToTensor(),  # Convert the image to a tensor,
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Normalize the pixel values
        ]
    )

    # apply transforms to the image
    x = transform(image)

    # print(f"Image shape: {x.shape}")

    # Add batch dimension
    x = x.unsqueeze(0)

    return x


# Convert image to tensor
x = img_to_tensor()

# # Initialize model
model = VitModel()

# Forward pass
out = model(x)

# Print output shape
print(out)
