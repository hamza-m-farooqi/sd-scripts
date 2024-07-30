import os
import re
import select
import shutil
import subprocess
import server_settings
import requests
from threading import Thread
from datetime import datetime
from decouple import config
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from request_queue import Job,TrainingRequest,TrainingConfig,JobStatus,TrainingResponse,job_queue
from server_logging import logger
from s3_utils import upload_media_to_s3,get_uploaded_media_from_s3



def background_training(job:Job):
    toml_path = os.path.join(job.job_config.output_dir, "config.toml")
    config:TrainingConfig=job.job_config
    project_deployment_path=server_settings.PROJECT_DEPLOYMENT_PATH
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
                        job.job_progress = percentage
                        process_logs(job,output)
                if fd == process.stderr.fileno():
                    read = process.stderr.readline()
                    if read:
                        output = read.strip()
                        print(output)
                        percentage=0
                        if f"/{config.max_train_steps}" in output:
                            percentage_value = get_progress_percentage(output)
                            percentage = percentage_value if percentage_value else 0
                        job.job_progress = percentage
                        process_logs(job,output)
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

    current_epoch = get_current_epoch(log)
    if current_epoch:
        epoch_response=TrainingResponse(total_epochs=job.job_epochs,current_epoch_number=current_epoch)
        job.job_results.append(epoch_response)
    saved_checkout_path = get_checkpoint_path(log)
    if saved_checkout_path:
        lora_file_name=os.path.basename(saved_checkout_path)
        epoch_model_s3_path=f"{job.job_s3_folder}{lora_file_name}"
        job.job_results[-1].epoch_model_s3_path=epoch_model_s3_path
        # job.job_results[-1].epoch_model_s3_url=get_uploaded_media_from_s3(epoch_model_s3_path)
        Thread(target=upload_media_to_s3,args=(saved_checkout_path,epoch_model_s3_path)).start()
        Thread(target=send_response,args=(job,)).start()
        
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

def get_total_epochs(log):
    # Updated regex pattern to match different possible formats
    pattern = r"num epochs.*?[:：]\s*(\d+)"
    match = re.search(pattern, log, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return None

def get_current_epoch(log):
    pattern = r"epoch\s*(\d+)/\d+"
    match = re.search(pattern, log, re.IGNORECASE)
    if match:
        return int(match.group(1))
    else:
        return None
    
def get_checkpoint_path(log):
    pattern = r"saving checkpoint:\s*(\/[^\s]+)"
    match = re.search(pattern, log, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        None

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

def send_response(job:Job):
    if not job.job_request.webhook_url:
        return
    requests.post(job.job_request.webhook_url, json=job.job_results[-1].dict())


