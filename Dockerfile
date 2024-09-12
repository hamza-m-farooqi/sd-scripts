# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables to avoid issues with prompts
ENV HF_TOKEN=""


# Install apt packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libomp-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    libgl1-mesa-glx \
    python3-pip \
    gdb \
    valgrind \
    git

# Update pip to the latest version
RUN python3 -m pip install --upgrade pip

# Copy all files into the /app directory
COPY . /app

# Set the working directory
WORKDIR /app

# Install Python packages
RUN python3 -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
RUN python3 -m pip install -r requirements-new.txt
RUN python3 -m pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

# Copy accelerate config file to the proper location
COPY default_config.yaml /root/.cache/huggingface/accelerate/default_config.yaml
