import os
import json
import requests
from PIL import Image
from io import BytesIO

ANNOTATIONS_FILE = "instances_val2017.json"  # Path to the annotations file
IMAGES_DIR = "coco_images"
NUMBER_OF_IMAGES_PER_RES = 10  #  images per resolution category


RESOLUTIONS = {
    "32x32": (32, 32),
    "64x64": (64, 64),
    "128x128": (128, 128),
    "256x256": (256, 256),
    "512x512": (512, 512),
    "1024x1024": (1024, 1024)
}


os.makedirs(IMAGES_DIR, exist_ok=True)


def load_annotations(file_path):
    with open(file_path, 'r') as f:
        annotations = json.load(f)
    return annotations


def resize_and_save_image(image_info, resolution, size):
    img_url = image_info['coco_url']
    img_filename = os.path.join(
        IMAGES_DIR, resolution, image_info['file_name'])

    try:
        response = requests.get(img_url)
        img = Image.open(BytesIO(response.content))

        img_resized = img.resize(size, Image.Resampling.LANCZOS)

        img_resized.save(img_filename)
        print(f"Saved image as {resolution}/{image_info['file_name']}")
    except Exception as e:
        print("Failed to download or resize image" +
              f"{image_info['file_name']}: {e}")


def download_and_resize_images(images, resolution, size):
    os.makedirs(os.path.join(IMAGES_DIR, resolution), exist_ok=True)
    for i, image_info in enumerate(images):
        print(f"Processing {resolution} image {i + 1}/{len(images)}")
        resize_and_save_image(image_info, resolution, size)


if not os.path.exists(ANNOTATIONS_FILE):
    print("Annotations file not found. Please download it from COCO dataset and place it at: " +
          f"{ANNOTATIONS_FILE}")
else:
    print("Loading COCO annotations...")
    annotations = load_annotations(ANNOTATIONS_FILE)

    all_images = annotations['images'][:NUMBER_OF_IMAGES_PER_RES *
                                       len(RESOLUTIONS)]

    for resolution, size in RESOLUTIONS.items():
        print(f"Starting to process images for resolution {resolution}")
        download_and_resize_images(
            all_images[:NUMBER_OF_IMAGES_PER_RES], resolution, size)
        all_images = all_images[NUMBER_OF_IMAGES_PER_RES:]

    print("Downloaded and resized images to " +
          f"{IMAGES_DIR} in fixed resolutions.")
