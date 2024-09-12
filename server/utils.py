import os
import uuid
import requests
from io import BytesIO
from PIL import Image
from server import server_settings


def create_folders_and_save_images(image_urls, lora_name, repeats):
    request_id = str(uuid.uuid4())
    base_path = f"{server_settings.LORA_TRAINING_DATASET_PATH}{request_id}_{lora_name}"
    images_path = os.path.join(base_path, "images")
    images_save_path = os.path.join(images_path, f"{repeats}_{lora_name}")
    models_path = os.path.join(base_path, "models")
    logs_path = os.path.join(base_path, "logs")

    # Ensure the directories exist
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    # Download and save each image
    for idx, url in enumerate(image_urls):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors
            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")
            image_path = os.path.join(images_save_path, f"{idx}.jpg")
            image.save(image_path, format="JPEG")
            print(f"Downloaded and saved image {idx} from {url} to {image_path}")
        except requests.RequestException as e:
            print(f"Failed to download {url}: {e}")
        except IOError as e:
            print(f"Failed to process image from {url}: {e}")

    print(f"Downloaded and saved {len(image_urls)} images to {images_save_path}")
    return base_path, images_path, images_save_path, models_path, logs_path
