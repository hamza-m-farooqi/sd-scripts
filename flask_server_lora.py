import os
import re
import uuid
import toml
import threading
import subprocess
import select
import shutil
from typing import List
from io import BytesIO
from flask import Flask, request, jsonify,send_file
from pydantic import BaseModel, Field, conint
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from decouple import config
app = Flask(__name__)


PRETRAINED_SDXL_MODEL_PATH=config("PRETRAINED_SDXL_MODEL_PATH")
LORA_TRAINING_DATASET_PATH=config("LORA_TRAINING_DATASET_PATH")
PROJECT_DEPLOYMENT_PATH=config("PROJECT_DEPLOYMENT_PATH")
TRAINED_LORA_FINAL_DESTINATION=config("TRAINED_LORA_FINAL_DESTINATION")
# Pydantic Model for Request
class TrainingRequest(BaseModel):
    lora_name: str
    learning_rate: float = 4e-7
    max_training_steps: conint(gt=0) = 1600
    network_dim: conint(gt=0) = 128
    network_alpha: conint(gt=0) = 64

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
    gradient_accumulation_steps: int = 6
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
    pretrained_model_name_or_path: str = PRETRAINED_SDXL_MODEL_PATH
    # pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
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
    base_path = f"{LORA_TRAINING_DATASET_PATH}{request_id}_{lora_name}"
    images_path = os.path.join(base_path, "images")
    images_save_path= os.path.join(images_path,f"100_{lora_name}")
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

def get_progress_percentage(output_line):
    # Regex pattern to find the diffusion process progress
    pattern = re.compile(r'(\d+)/(\d+)\s+\[')
    match = pattern.search(output_line)
    print(match)
    if match:
        current, total = map(int, match.groups())
        percentage = (current / total) * 100
        return percentage
    return None


def background_training(toml_path,config:Config,training_request:TrainingRequest):
    project_deployment_path=PROJECT_DEPLOYMENT_PATH
    # command = f"bash -c 'source {project_deployment_path}/venv/bin/activate && cd {project_deployment_path}/  && python sdxl_train.py --config {toml_path}'"
    command = f"bash -c 'source {project_deployment_path}/venv/bin/activate && cd {project_deployment_path}/  && accelerate launch --dynamo_backend no --dynamo_mode default --mixed_precision fp16 --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 2 sdxl_train_network.py --config {toml_path}'"
    print(command)

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Use select to read from stdout and stderr without blocking
        while True:
            reads = [process.stdout.fileno(), process.stderr.fileno()]
            ret = select.select(reads, [], [])
            for fd in ret[0]:
                if fd == process.stdout.fileno():
                    read = process.stdout.readline()
                    if read:
                        output = read.strip()
                        print(output)
                        percentage=0
                        if f"/{config.max_train_steps}"  in output:
                            percentage_value = get_progress_percentage(output)
                            percentage = percentage_value if percentage_value else 0
                        # job.job_progress = percentage
                if fd == process.stderr.fileno():
                    read = process.stderr.readline()
                    if read:
                        output = read.strip()
                        print(output)
                        percentage=0
                        if f"/{config.max_train_steps}" in output:
                            percentage_value = get_progress_percentage(output)
                            percentage = percentage_value if percentage_value else 0
                        # job.job_progress = percentage

            if process.poll() is not None:
                break

        return_code = process.poll()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)

        print("Job is Finished")
        trained_model_path=os.path.join(config.output_dir,f"{config.output_name}.{config.save_model_as}")
        if not os.path.exists(trained_model_path):
            print("Trained Model Not Found!")
        # MOVE TRAINED LORA MODEL FROM trained_model_path to TRAINED_LORA_FINAL_DESTINATION
        shutil.move(trained_model_path, TRAINED_LORA_FINAL_DESTINATION)
        print("Trained Lora Model Moved to: ", TRAINED_LORA_FINAL_DESTINATION)

        # job.job_status = "finished"
        # job.job_result = os.path.join(
        #     settings.STATIC_URL,
        #     "viton",
        #     "outputs",
        #     os.path.basename(job_params.output_path),
        # )

    except subprocess.CalledProcessError as e:
        print(e)
        # job.job_status = "failed"
        # job.error_message = str(e)

    except Exception as e:
        print(e)
        # job.job_status = "failed"
        # job.error_message = str(e)

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
        output_name=lora_name,
        logging_dir=logs_path,
        sample_prompts=os.path.join(models_path, "prompt.txt")
    )

    toml_path = os.path.join(models_path, "config.toml")
    with open(toml_path, "w") as toml_file:
        toml.dump(config.dict(), toml_file)

    # Run training command
    threading.Thread(target=background_training,args=(toml_path,config,training_request)).start()
    return jsonify({"status": "Training started", "config": config.dict()}), 200

@app.route('/download', methods=['GET'])
def download():
    path = request.args.get('path')
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
