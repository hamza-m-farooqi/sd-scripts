import os
from threading import Thread
import uuid
import toml
import json
from server import server_settings
from typing import List
from io import BytesIO
from PIL import Image
from flask import Flask, request, jsonify, send_file
from decouple import config
from server.request_queue import Job, TrainingRequest, TrainingConfig, JobQueue,SDModel, job_queue
from server.request_worker import viton_task_loop
app = Flask(__name__)


# Create necessary folders and save images
def create_folders_and_save_images(files, lora_name, repeats):
    request_id = str(uuid.uuid4())
    base_path = f"{server_settings.LORA_TRAINING_DATASET_PATH}{request_id}_{lora_name}"
    images_path = os.path.join(base_path, "images")
    images_save_path = os.path.join(images_path, f"{repeats}_{lora_name}")
    models_path = os.path.join(base_path, "models")
    logs_path = os.path.join(base_path, "logs")

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    for idx, file in enumerate(files):
        image = Image.open(BytesIO(file.read()))
        image = image.convert("RGB")
        image_path = os.path.join(images_save_path, f"{idx}.jpg")
        image.save(image_path, format="JPEG")

    return base_path, images_path, images_save_path, models_path, logs_path


# Generate captions using BLIP


@app.route("/train", methods=["POST"])
def train():
    training_request_defaults = TrainingRequest()
    lora_name = request.form["lora_name"]
    sd_model = request.form.get("sd_model",training_request_defaults.sd_model)
    learning_rate = float(
        request.form.get("learning_rate", training_request_defaults.learning_rate)
    )
    max_train_epochs = request.form.get(
            "max_train_epochs", training_request_defaults.max_train_epochs
        )
    max_training_steps = request.form.get(
            "max_training_steps", training_request_defaults.max_training_steps
        )
    network_dim = int(
        request.form.get("network_dim", training_request_defaults.network_dim)
    )
    network_alpha = int(
        request.form.get("network_alpha", training_request_defaults.network_alpha)
    )
    repeats = int(request.form.get("repeats", training_request_defaults.repeats))
    example_prompts = request.form.get(
        "example_prompts", training_request_defaults.example_prompts
    )
    example_prompts = [] if not example_prompts else json.loads(example_prompts)
    webhook_url = str(request.form.get("webhook_url", training_request_defaults.webhook_url))

    files = request.files.getlist("images")

    if len(files) == 0:
        return jsonify({"error": "No images uploaded"}), 400

    training_request = TrainingRequest(
        lora_name=lora_name,
        max_train_epochs=max_train_epochs,
        learning_rate=learning_rate,
        max_training_steps=max_training_steps,
        network_dim=network_dim,
        network_alpha=network_alpha,
        webhook_url=webhook_url,
        example_prompts=example_prompts,
        sd_model=sd_model,
        repeats=repeats
        
    )

    base_path, images_path, images_save_path, models_path, logs_path = (
        create_folders_and_save_images(files, training_request.lora_name, repeats)
    )

    # generate_captions(images_save_path, training_request.lora_name)
    sample_prompts_file_path = os.path.join(models_path, "prompt.txt")
    # create empty prompt.txt file if example prompts is empty list else write each prompt in new line in file
    # if not example_prompts:
    #     with open(sample_prompts_file_path, "w") as f:
    #         pass
    # else:
    #     with open(sample_prompts_file_path, "w") as f:
    #         f.write("\n".join(example_prompts))

    config = TrainingConfig(

        learning_rate=training_request.learning_rate,
        max_train_steps=training_request.max_training_steps,
        max_train_epochs=training_request.max_train_epochs,
        network_dim=training_request.network_dim,
        network_alpha=training_request.network_alpha,
        train_data_dir=images_path,
        output_dir=models_path,
        output_name=lora_name,
        logging_dir=logs_path,
        sample_prompts=sample_prompts_file_path,
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" if training_request.sd_model==SDModel.SD_1_5.value else server_settings.PRETRAINED_SDXL_MODEL_PATH,
    )
    
    if training_request.sd_model==SDModel.SDXL_1_0.value:
        config.train_batch_size=1
    
    total_images=(len(files)*training_request.repeats)
    total_images=total_images/config.train_batch_size
    total_training_steps = total_images*training_request.max_train_epochs
    training_request.total_steps=int(total_training_steps)

    toml_path = os.path.join(models_path, "config.toml")
    with open(toml_path, "w") as toml_file:
        toml.dump(config.dict(), toml_file)

    job = Job(job_request=training_request, job_config=config)
    job.job_number = len(job_queue.pending) + 1

    # Add the job to the queue
    job_queue.pending.append(job)
    return jsonify(job.dict()), 200


@app.route("/download", methods=["GET"])
def download():
    path = request.args.get("path")
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


@app.route("/train", methods=["GET"])
def training_get():
    job_id = request.args.get("job_id")
    for job in job_queue.pending:
        if job.job_id == job_id:
            return jsonify(job.dict()), 200
        # Check in the history
    for job in job_queue.history:
        if job.job_id == job_id:
            return jsonify(job.dict()), 200

    return jsonify({"error": "Job ID not found"}), 404


Thread(target=viton_task_loop).start()
if __name__ == "__main__":
    print("Going to run flask server")
    app.run(host="0.0.0.0", port=7000)
