from PIL import Image
import io
import numpy as np
import torch

def inpainting_score(inpainted_image, mask, original_image):
    # Ensure images, original_image, and masks are in the right format
    if isinstance(inpainted_image, torch.Tensor):
        inpainted_image = inpainted_image.cpu().numpy()

    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()

    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()


    # Initialize a variable to store the cumulative L1 distance
    total_distance = 0.0

    for i, m, o in zip(inpainted_image, mask, original_image):
        # Calculate L1 distance for pixels where the mask is 0
        distance = np.abs(o - i) * (1 - m)

        # Sum the L1 distances for this image
        total_distance += distance.mean()

    # Calculate the score (negative L1 distance) for reinforcement learning
    score = -total_distance

    return np.array([score])


def jpeg_incompressibility(images, masks):
    if isinstance(images, torch.Tensor):
        images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

    masked_images = []
    for mask, image in zip(masks, images):
        mask = mask.round().cpu().numpy().transpose(1, 2, 0)
        masked_images.append(image * mask)
    images = [image * mask for image in images]
    images = [Image.fromarray(image.astype(np.uint8)) for image in images]
    buffers = [io.BytesIO() for _ in images]
    for image, buffer in zip(images, buffers):
        image.save(buffer, format="JPEG", quality=95)
    sizes = [buffer.tell() / 1000 for buffer in buffers]
    return np.array(sizes)

def jpeg_compressibility(images, masks):
    rew = jpeg_incompressibility(images, masks)
    return -rew


def aesthetic_score(scorer, images, masks):
    images = (images * 255).round().clamp(0, 255).to(torch.uint8)
    scores = scorer(images)
    return scores