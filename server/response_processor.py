import os
import copy
import requests
from threading import Thread
from server.request_queue import Job,TrainingResponse
from server.s3_utils import upload_to_s3

def process_response(job:Job,safetensors_files:set):
    current_files = {f for f in os.listdir(job.job_config.output_dir) if f.endswith('.safetensors')}
    new_files = current_files - safetensors_files
    if new_files:
        for new_file in new_files:
            print(f"New file found: {new_file}")
            epoch_response=TrainingResponse(total_epochs=job.job_epochs,current_epoch_number=len(job.job_results)+1)
            epoch_model_s3_path=f"{job.job_s3_folder}{new_file}"
            saved_checkout_path=os.path.join(job.job_config.output_dir,new_file)
            epoch_response.epoch_model_s3_path=epoch_model_s3_path
            epoch_response.epoch_model_s3_url=upload_to_s3(saved_checkout_path,epoch_model_s3_path)
            print("Model Uploaded in S3 at ",epoch_response)
            print("Local Path of uploaded model is ",saved_checkout_path)
            job.job_results.append(epoch_response)
            send_response(job)
        safetensors_files.update(new_files)

          

def send_response(job:Job):
    if not job.job_request.webhook_url or job.job_request.webhook_url=="" or job.job_request.webhook_url=='None':
        return
    webhook_response(job.job_request.webhook_url,True,200, "Success", job.dict())

def webhook_response(webhook_url, status, code, message, data=None):
    def send(webhook_url, status, code, message, data=None):
        response_data = {
            "status": status,
            "code": code,
            "message": message,
            "data": data,
        }
        print("Going to send data over webhook!")
        print(response_data)
        if webhook_url and "http" in webhook_url:
            requests.post(webhook_url, json=response_data)

    Thread(target=send, args=(webhook_url, status, code, message, data)).start()
    return None