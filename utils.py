import random
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.path as path
import os
from roboflow import Roboflow

def set_random_seed(random_seed: int = 42) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    cv2.setRNGSeed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def download_dataset_folder(version_number, api_key=None):
    if api_key is None:
        api_key = os.environ.get("ROBOFLOW_API_KEY")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("arked").project("white-facades")
    dataset = project.version(1).download("yolov7")

import matplotlib.pyplot as plt
import matplotlib.path as path
import numpy as np
import os
import cv2
import yaml

# Function to parse the text label and extract polygon information
def parse_labels(label_file):
    with open(label_file, 'r') as file:
        lines = file.readlines()

    polygons = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(parts[0])
            coordinates = [float(coord) for coord in parts[1:]]
            num_points = len(coordinates) // 2  # Number of (x, y) pairs
            polygon = [(coordinates[i], coordinates[i + 1]) for i in range(0, len(coordinates), 2)]
            polygons.append((class_id, polygon))

    return polygons

# Function to plot bounding boxes
def plot_polygons(image_path, polygons):
    # Load the image if needed
    image = plt.imread(image_path)

    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Generate unique colors for classes
    class_ids = set([class_id for class_id, _ in polygons])
    num_classes = len(class_ids)
    colors = generate_colors(num_classes)

    # Plot each polygon
    for class_id, polygon in polygons:
        x, y = zip(*polygon)
        color = colors[class_id % num_classes]
        polygon = plt.Polygon(np.c_[x, y], linewidth=1, edgecolor=color, facecolor='none', label=f'Class {class_id}')
        ax.add_patch(polygon)

    # Display the image
    ax.imshow(image)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.legend()

    # Show the plot
    plt.show()

# Function to generate unique colors based on class ID
def generate_colors(num_classes):
    colormap = plt.cm.get_cmap('tab20', num_classes)
    colors = [colormap(i) for i in range(num_classes)]
    return colors

# Function to scale polygons to match an image size
def scale_polygons(polygons, image_size):
    scaled_polygons = []

    for class_id, polygon in polygons:
        # Scale the polygon coordinates
        scaled_polygon = [(x * image_size[0], y * image_size[1]) for x, y in polygon]
        scaled_polygons.append((class_id, scaled_polygon))

    return scaled_polygons

def enlarge_mask(mask, scaling_pixels=1):
    enlarged_mask = mask.copy()

    for s in range(1, scaling_pixels+1):
        # Create masks for shifting in all four directions
        up_mask = np.roll(mask, s, axis=0)
        down_mask = np.roll(mask, -s, axis=0)
        left_mask = np.roll(mask, s, axis=1)
        right_mask = np.roll(mask, -s, axis=1)

        # Use logical OR to combine the shifted masks
        enlarged_mask = np.logical_or(enlarged_mask, up_mask)
        enlarged_mask = np.logical_or(enlarged_mask, down_mask)
        enlarged_mask = np.logical_or(enlarged_mask, left_mask)
        enlarged_mask = np.logical_or(enlarged_mask, right_mask)

    return enlarged_mask.astype(np.uint8)

# generate mask from polygons
def generate_masks(polygons, image_size, num_samples=1, scaling_pixels=None, labels_filter=None):
    mask = np.zeros(image_size, dtype=np.uint8)
    i=0
    class_id = None
    for class_id, polygon in polygons:
        if (labels_filter is None) or (class_id in labels_filter):
            x, y = zip(*polygon)
            # Create a Path object from the polygon coordinates
            path_polygon = path.Path(list(zip(x, y)))

            # Generate a mask for the polygon
            x, y = np.meshgrid(np.arange(image_size[0]), np.arange(image_size[1]))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            mask_indices = path_polygon.contains_points(points).reshape(image_size[0], image_size[1])
            # Fill the polygon in the mask
            mask[mask_indices] = 1

            i+=1
            if (i == num_samples):
                break

    if scaling_pixels:
        mask = enlarge_mask(mask, scaling_pixels)

    return mask, class_id

def create_dataset(image_paths, label_paths, num_samples=1, resize_shape=None, scaling_pixels=None, labels_filter=None):
    images = []
    masks = []
    labels = []

    # list of all the images in the image folder
    images_names = os.listdir(image_paths)
    label_names = [image_name.replace(".jpg", ".txt") for image_name in images_names]
    for img_name, label_name in zip(images_names, label_names):
        if (label_name not in os.listdir(label_paths)):
            images_names.remove(img_name)
            label_names.remove(label_name)
        img_path = os.path.join(image_paths, img_name)
        label_path = os.path.join(label_paths, label_name)

        # read the rgb image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_size = (img.shape[0], img.shape[1])
        polygons = parse_labels(label_path)
        scaled_polygons = scale_polygons(polygons, image_size)
        mask, label = generate_masks(scaled_polygons, image_size, num_samples=num_samples, scaling_pixels=scaling_pixels, labels_filter=labels_filter)

        # resize the image
        if resize_shape is not None:
            img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, resize_shape, interpolation=cv2.INTER_AREA)

        if len(np.unique(mask)) > 1:
            images.append(img)
            masks.append(mask)
            labels.append(label)

    return np.array(images), np.array(masks), np.array(labels)

# mask the images with the masks
def mask_images(images, masks, invert=False):
    masked_images = []
    for img, mask in zip(images, masks):
        if invert:
            masked_images.append(img * (1 - mask[..., np.newaxis]))
        else:
            masked_images.append(img * mask[..., np.newaxis])
    return np.array(masked_images)

def get_label_mapping(data_yaml):
    with open(data_yaml, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    label_mapping = {}
    for i, label in enumerate(data['names']):
        label_mapping[i] = label

    return label_mapping