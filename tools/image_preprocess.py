from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

# import struct
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

# Convert the image to RGB565 format
def convert_to_rgb565(image):
    image_np = np.array(image)
    r = (image_np[:, :, 0] >> 3).astype(np.uint16)
    g = (image_np[:, :, 1] >> 2).astype(np.uint16)
    b = (image_np[:, :, 2] >> 3).astype(np.uint16)
    rgb565 = (r << 11) | (g << 5) | b
    return rgb565

# Save the RGB565 image data to a file
def save_rgb565(image, filename):
    rgb565 = convert_to_rgb565(image)
    rgb565.tofile(filename)


def load_raw_rgb565(filename, width, height):
    rgb565 = np.fromfile(filename, dtype=np.uint16).reshape((height, width))
    r = ((rgb565 >> 11) & 0x1F) << 3
    g = ((rgb565 >> 5) & 0x3F) << 2
    b = (rgb565 & 0x1F) << 3
    image_np = np.stack((r, g, b), axis=-1).astype(np.uint8)
    return Image.fromarray(image_np)


# Load image
image = Image.open("./demo/resnet18/apple.jpg")
print("Image size: {}, width: {}, height: {}".format(image.size, image.width, image.height))

# Save the image as RGB565
save_rgb565(image, "./demo/resnet18/apple.rgb")

# Load the image from the RGB565 file
loaded_image = load_raw_rgb565("./demo/resnet18/apple.rgb", 730, 730)

# Save the loaded image to verify
# loaded_image.save("./demo/resnet18/apple.png")

# Define the preprocessing operations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
])

# Apply preprocessing operations
preprocessed_image = transform(loaded_image)

# Convert the preprocessed image to a NumPy array
preprocessed_image_np = preprocessed_image.numpy()
np.save('./demo/resnet18/apple_after_resize.npy', preprocessed_image_np)

# Convert the preprocessed image data to int8
preprocessed_image_int8 = np.floor(preprocessed_image_np * 64 + 0.5).astype(np.int8)
preprocessed_image_int8.tofile("./demo/resnet18/apple_after_resize.bin")

# denormalize_test = False
denormalize_test = True

def denormalize(tensor, mean, std):
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    denormalized_tensor = tensor * std_tensor + mean_tensor
    return denormalized_tensor.clamp(0, 1)  # 将值限制在 [0, 1] 范围内

if denormalize_test:

    denormalized_image = denormalize(preprocessed_image, CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD)
    # denormalized_image = preprocessed_image

    denormalized_image_np = (denormalized_image.numpy() * 255).astype(np.uint8)
    denormalized_image_pil = Image.fromarray(denormalized_image_np.transpose(1, 2, 0))

    denormalized_image_pil.save("./demo/resnet18/apple_after_resize.png")

print("Preprocess done!")