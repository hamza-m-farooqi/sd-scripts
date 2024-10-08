import os
import json
import toml
import argparse
import requests as http_requests
import server.server_settings as server_settings
from server.request_queue import (
    Job,
    TrainingRequest,
    TrainingRequest_SD15,
    TrainingRequest_SDXL,
    TrainingConfig,
    SDModel,
)
from server.utils import create_folders_and_save_images
from server.request_processor import background_training


# /home/ubuntu/.cache/huggingface/accelerate/default_config.yaml 

def train(training_request_dict: dict):
    try:
        if training_request_dict.get("base_model") == SDModel.SD_1_5.value:
            training_request_defaults = TrainingRequest_SD15()
        else:
            training_request_defaults = TrainingRequest_SDXL()
        job_id = training_request_dict.get("job_id")
        lora_name = training_request_dict.get("lora_name")
        sd_model = training_request_dict.get("base_model", training_request_defaults.sd_model)
        learning_rate = float(
            training_request_dict.get(
                "learning_rate", training_request_defaults.learning_rate
            )
        )
        max_train_epochs = training_request_dict.get(
            "max_train_epochs", training_request_defaults.max_train_epochs
        )
        max_training_steps = training_request_dict.get(
            "max_training_steps", training_request_defaults.max_training_steps
        )
        network_dim = int(
            training_request_dict.get("network_dim", training_request_defaults.network_dim)
        )
        network_alpha = int(
            training_request_dict.get(
                "network_alpha", training_request_defaults.network_alpha
            )
        )
        repeats = int(
            training_request_dict.get("repeats", training_request_defaults.repeats)
        )
        example_prompts = training_request_dict.get(
            "example_prompts", training_request_defaults.example_prompts
        )
        example_prompts = [] if not example_prompts else json.loads(example_prompts)

        webhook_url = str(
            training_request_dict.get("webhook_url", training_request_defaults.webhook_url)
        )

        images_urls = training_request_dict.get("images_urls", [])


        if not job_id:
            return webhook_response(webhook_url, False, 400, "No job id provided!")
        if not lora_name:
            return webhook_response(webhook_url, False, 400, "No lora name provided!")
        if len(images_urls) == 0:
            return webhook_response(webhook_url, False, 400, "No image urls provided!")
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
            repeats=repeats,
        )
        base_path, images_path, images_save_path, models_path, logs_path = (
            create_folders_and_save_images(images_urls, training_request.lora_name, repeats)
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
            pretrained_model_name_or_path=(
                "/app/models/epicrealism.safetensors"
                if training_request.sd_model == SDModel.SD_1_5.value
                else server_settings.PRETRAINED_SDXL_MODEL_PATH
            ),
        )

        if training_request.sd_model == SDModel.SDXL_1_0.value:
            config.train_batch_size = 1
            config.lr_scheduler = "cosine"
            config.text_encoder_lr = 5e-5
            config.unet_lr = 1e-4
            config.lr_warmup_steps = 0
            config.bucket_no_upscale = False
            
        total_images = len(images_urls) * training_request.repeats
        total_images = total_images / config.train_batch_size
        total_training_steps = total_images * training_request.max_train_epochs
        training_request.total_steps = int(total_training_steps)

        toml_path = os.path.join(models_path, "config.toml")
        with open(toml_path, "w") as toml_file:
            toml.dump(config.dict(), toml_file)

        job = Job(job_id=job_id,job_request=training_request, job_config=config)
        job.job_number = 1

        # Add the job to the queue
        background_training(job)
    except Exception as ex:
        print(ex)
        webhook_response(webhook_url, False, 500, str(ex))
        raise Exception(ex)


def webhook_response(webhook_url, status, code, message, data=None):
    response_data = {"status": status, "code": code, "message": message, "data": data}
    print(response_data)
    if webhook_url and "http" in webhook_url:
        http_requests.post(webhook_url, json=response_data)
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Training Job")
    parser.add_argument('--training_request', type=str, required=True, help='Training request JSON string')
    return parser.parse_args()
 

if __name__ == "__main__":
    args = parse_args()
    # Convert the JSON string back to a dictionary
    training_request_dict = json.loads(args.training_request)
    
    # Call the train function with the parsed dictionary
    train(training_request_dict)
# training_request_dict={
#     "lora_name": "Irfan_New",
#     "sd_model": "SDXL 1.0",
#     "webhook_url":"https://webhook-test.com/f447d103cb690b7a656a61306cec7b23",
#     "images_urls":[
#             "https://boothybooth.s3.amazonaws.com/lora_images/Savaiz/63~2_NNNLJLT.png",
#             "https://boothybooth.s3.amazonaws.com/lora_images/Savaiz/IMG_2259~3_316EIoA.jpg",
#             "https://boothybooth.s3.amazonaws.com/lora_images/Savaiz/IMG_3002_0cpbQyN.HEIC",
#             "https://boothybooth.s3.amazonaws.com/lora_images/Savaiz/IMG_3003_amZKwYh.HEIC",
#             "https://boothybooth.s3.amazonaws.com/lora_images/Savaiz/IMG_5615~2_Vy45QWe.jpg",
#             "https://boothybooth.s3.amazonaws.com/lora_images/Savaiz/IMG_6931_lugTWL6.jpg",
#             "https://boothybooth.s3.amazonaws.com/lora_images/Savaiz/IMG_6932_qgKZjWe.jpg",
#             "https://boothybooth.s3.amazonaws.com/lora_images/Savaiz/IMG20240126113742_9LPQYOx.jpg",
#             "https://boothybooth.s3.amazonaws.com/lora_images/Savaiz/PXL_20240716_162039748.MP_kkAKpte.jpg",
#             "https://boothybooth.s3.amazonaws.com/lora_images/Savaiz/PXL_20240716_163147619.NIGHT_FBwvJSG.jpg"
#     ],
#     "job_id":"1234"
#  }
# train(training_request_dict)