import uuid
import server_settings
from enum import Enum
from datetime import datetime
from typing import List
from pydantic import BaseModel, conint
class JobStatus(Enum):
    WAITING = "waiting"
    PROCESSING = "processing"
    FINISHED = "finished"
    FAILED = "failed"

class TrainingRequest(BaseModel):
    lora_name: str=""
    repeats: int = 100
    learning_rate: float = 4e-7
    max_training_steps: conint(gt=0) = 10
    network_dim: conint(gt=0) = 128
    network_alpha: conint(gt=0) = 64
    example_prompts:list=None
    webhook_url:str = None

class TrainingResponse(BaseModel):
    total_epochs : int = 0
    current_epoch_number : int = 0
    current_epoch_id:str = str(uuid.uuid4())
    epoch_model_s3_path : str = ""
    epoch_model_s3_url : str = ""


class TrainingConfig(BaseModel):
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
    pretrained_model_name_or_path: str = server_settings.PRETRAINED_SDXL_MODEL_PATH
    # pretrained_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    prior_loss_weight: int = 1
    resolution: str = "1024,1024"
    sample_prompts: str = ""
    sample_sampler: str = "euler_a"
    # sample_every_n_epochs: int = 1
    save_every_n_epochs: int = 1
    save_model_as: str = "safetensors"
    save_precision: str = "bf16"
    text_encoder_lr: float = 4e-7
    train_batch_size: int = 1
    train_data_dir: str = ""
    unet_lr: float = 0.0001
    xformers: bool = True

class Job(BaseModel):
    job_id:str = str(uuid.uuid4())
    job_request:TrainingRequest = None
    job_config:TrainingConfig = None
    job_number:int = None  # This will be set when added to the queue
    job_progress:int = 0
    job_status:str = JobStatus.WAITING.value  # Initial status
    job_epochs:int = 0
    job_s3_folder:str=f"loras/{datetime.now().strftime('%Y-%m-%d')}/{str(uuid.uuid4())}/"
    job_results:List[TrainingResponse] = []
    error_message:str = None


class JobQueue:
    def __init__(self):
        self.pending: List[Job] = []
        self.history: List[Job] = []
        self.last_job_id = None

job_queue = JobQueue()

