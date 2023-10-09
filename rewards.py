from PIL import Image
import io
import numpy as np
import torch


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