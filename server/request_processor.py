import os
import re
import select
import shutil
import subprocess
from server import server_settings
import requests
from threading import Thread
from datetime import datetime
from decouple import config
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from server.request_queue import Job,TrainingRequest,TrainingConfig,JobStatus,TrainingResponse,SDModel,job_queue
from server.server_logging import logger
from server.s3_utils import upload_media_to_s3,get_uploaded_media_from_s3
from server.response_processor import process_response


def background_training(job:Job):
    toml_path = os.path.join(job.job_config.output_dir, "config.toml")
    config:TrainingConfig=job.job_config
    project_deployment_path=server_settings.PROJECT_DEPLOYMENT_PATH
    # command = f"bash -c 'source {project_deployment_path}/venv/bin/activate && cd {project_deployment_path}/  && python sdxl_train.py --config {toml_path}'"
    if job.job_request.sd_model==SDModel.SDXL_1_0.value:
        command = f"bash -c 'source {project_deployment_path}/venv/bin/activate && cd {project_deployment_path}/  && accelerate launch --dynamo_backend no --dynamo_mode default --mixed_precision fp16 --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 2 sdxl_train_network.py --config {toml_path}'"
    else:
        command = f"bash -c 'source {project_deployment_path}/venv/bin/activate && cd {project_deployment_path}/  && accelerate launch --dynamo_backend no --dynamo_mode default --mixed_precision fp16 --num_processes 1 --num_machines 1 --num_cpu_threads_per_process 2 train_network.py --config {toml_path}'"
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
        logs_count=0
        safetensors_files = set()
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
                        if f"/{job.job_request.total_steps}"  in output:
                            percentage_value = get_progress_percentage(output)
                            percentage = percentage_value if percentage_value else 0
                            logs_count+=1
                        job.job_progress = percentage
                        process_logs(job,output)
                        if logs_count%20==0:
                            process_response(job,safetensors_files)
                if fd == process.stderr.fileno():
                    read = process.stderr.readline()
                    if read:
                        output = read.strip()
                        print(output)
                        percentage=0
                        if f"/{job.job_request.total_steps}" in output:
                            percentage_value = get_progress_percentage(output)
                            percentage = percentage_value if percentage_value else 0
                            logs_count+=1
                        job.job_progress = percentage
                        process_logs(job,output)
                        if logs_count%20==0:
                            process_response(job,safetensors_files)
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
        # shutil.move(trained_model_path, server_settings.TRAINED_LORA_FINAL_DESTINATION)
        print("Trained Lora Model Moved to: ", server_settings.TRAINED_LORA_FINAL_DESTINATION)
        job.job_progress = 100
        job.job_status = JobStatus.FINISHED.value


        # job.job_status = "finished"
        # job.job_result = os.path.join(
        #     settings.STATIC_URL,
        #     "viton",
        #     "outputs",
        #     os.path.basename(job_params.output_path),
        # )

    except subprocess.CalledProcessError as e:
        print(e)
        job.job_status=JobStatus.FAILED.value
        job.error_message = str(e)

    except Exception as e:
        print(e)
        job.job_status=JobStatus.FAILED.value
        job.error_message = str(e)
    finally:
        job_queue.history.append(job)
        job_queue.pending.remove(job)

def process_logs(job:Job,log):
    total_epochs = get_total_epochs(log)
    if total_epochs:
        job.job_epochs=total_epochs   

def get_progress_percentage(output_line):
    # Regex pattern to find the diffusion process progress
    pattern = re.compile(r'(\d+)/(\d+)\s+\[')
    match = pattern.search(output_line)
    if match:
        current, total = map(int, match.groups())
        percentage = (current / total) * 100
        return percentage
    return None

def get_total_epochs(log):
    # Updated regex pattern to match different possible formats
    pattern = r"num epochs.*?[:：]\s*(\d+)"
    match = re.search(pattern, log, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return None



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

def process_request(job:Job):
    logger.info(f"Going to Process Job: {job.job_id}")
    job.job_status=JobStatus.PROCESSING.value
    images_save_path=os.path.join(job.job_config.train_data_dir,f"{job.job_request.repeats}_{job.job_request.lora_name}")
    generate_captions(images_save_path, job.job_request.lora_name)
    background_training(job)



