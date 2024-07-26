from flask import Flask, request, jsonify
from pydantic import BaseModel, Field, conint
from typing import List
import os
import uuid
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import toml
import subprocess
from io import BytesIO

app = Flask(__name__)

# Pydantic Model for Request
class TrainingRequest(BaseModel):
    lora_name: str
    learning_rate: float = 4e-7
    max_training_steps: conint(gt=0) = 1600
    network_dim: conint(gt=0) = 128
    network_alpha: conint(gt=0) = 128

# Pydantic Model for Config
class Config(BaseModel):
    bucket_no_upscale: bool = True
    bucket_reso_steps: int = 64
    cache_latents: bool = True
    caption_extension: str = ".txt"
    clip_skip: int = 1
    dynamo_backend: str = "no"
    enable_bucket: bool = True
    epoch: int = 1
    gradient_accumulation_steps: int = 4
    huber_c: float = 0.1
    huber_schedule: str = "snr"
    learning_rate: float = 4e-7
    logging_dir: str = ""
    loss_type: str = "l2"
    lr_scheduler: str = "constant_with_warmup"
    lr_scheduler_args: list = []
    lr_scheduler_num_cycles: int = 1
    lr_scheduler_power: int = 1
    lr_warmup_steps: int = 160
    max_bucket_reso: int = 2048
    max_data_loader_n_workers: int = 0
    max_grad_norm: int = 1
    max_timestep: int = 1000
    max_token_length: int = 75
    max_train_steps: int = 1600
    min_bucket_reso: int = 256
    mixed_precision: str = "fp16"
    multires_noise_discount: float = 0.3
    network_alpha: int = 64
    network_args: list = []
    network_dim: int = 128
    network_module: str = "networks.lora"
    no_half_vae: bool = True
    noise_offset_type: str = "Original"
    optimizer_args: list = []
    optimizer_type: str = "AdamW8bit"
    output_dir: str = ""
    output_name: str = "last"
    pretrained_model_name_or_path: str = "/var/www/projects/FOOCUS-Django/REPOSITORIES/Fooocus/models/checkpoints/epicrealismXL_V7FinalDestination.safetensors"
    prior_loss_weight: int = 1
    resolution: str = "1024,1024"
    sample_prompts: str = ""
    sample_sampler: str = "euler_a"
    save_every_n_epochs: int = 1
    save_model_as: str = "safetensors"
    save_precision: str = "bf16"
    text_encoder_lr: float = 4e-7
    train_batch_size: int = 1
    train_data_dir: str = ""
    unet_lr: float = 0.0001
    xformers: bool = True

# Create necessary folders and save images
def create_folders_and_save_images(files, lora_name):
    request_id = str(uuid.uuid4())
    base_path = f"/var/www/projects/lora_datasets/{request_id}_{lora_name}"
    images_path = os.path.join(base_path, "images")
    images_save_path= os.path.join(images_path,f"100_{lora_name}.safetensors")
    models_path = os.path.join(base_path, "models")
    logs_path = os.path.join(base_path, "logs")

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    for idx, file in enumerate(files):
        image = Image.open(BytesIO(file.read()))
        image = image.convert('RGB')
        image_path = os.path.join(images_save_path, f"{idx}.jpg")
        image.save(image_path, format="JPEG")

    return base_path, images_path,images_save_path, models_path, logs_path

# Generate captions using BLIP
def generate_captions(images_path, lora_name):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    for image_file in os.listdir(images_path):
        image_path = os.path.join(images_path, image_file)
        image = Image.open(image_path)
        inputs = processor(image, return_tensors="pt")
        caption_ids = model.generate(**inputs)
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)
        caption = f"{lora_name} {caption}"
        caption_file = os.path.join(images_path, image_file.replace(".jpg", ".txt"))
        with open(caption_file, "w") as f:
            f.write(caption)

@app.route('/train', methods=['POST'])
def train():
    lora_name = request.form['lora_name']
    learning_rate = float(request.form.get('learning_rate', 4e-7))
    max_training_steps = int(request.form.get('max_training_steps', 1600))
    network_dim = int(request.form.get('network_dim', 128))
    network_alpha = int(request.form.get('network_alpha', 128))
    
    files = request.files.getlist('images')
    
    if len(files) == 0:
        return jsonify({"error": "No images uploaded"}), 400

    training_request = TrainingRequest(
        lora_name=lora_name,
        learning_rate=learning_rate,
        max_training_steps=max_training_steps,
        network_dim=network_dim,
        network_alpha=network_alpha
    )

    base_path, images_path,images_save_path, models_path, logs_path = create_folders_and_save_images(files, training_request.lora_name)
    
    generate_captions(images_save_path, training_request.lora_name)

    config = Config(
        learning_rate=training_request.learning_rate,
        max_train_steps=training_request.max_training_steps,
        network_dim=training_request.network_dim,
        network_alpha=training_request.network_alpha,
        train_data_dir=images_path,
        output_dir=models_path,
        logging_dir=logs_path,
        sample_prompts=os.path.join(models_path, "prompt.txt")
    )

    toml_path = os.path.join(models_path, "config.toml")
    with open(toml_path, "w") as toml_file:
        toml.dump(config.dict(), toml_file)

    # Run training command
    command = f"python train_network.py --config {toml_path}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def process_output(process):
        while True:
            output = process.stdout.readline()
            if output == b"" and process.poll() is not None:
                break
            if output:
                print(output.strip().decode())

    process_output(process)
    process.wait()

    return jsonify({"status": "Training started", "config": config.dict()}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
