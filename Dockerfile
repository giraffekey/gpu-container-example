FROM nvidia/cuda:12.6.0-base-ubuntu24.04

# Install Pip
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-venv

# Create virtual environment
RUN python3 -m venv workspace

# Install PyTorch and Transformers libraries
RUN /workspace/bin/pip install torch transformers

# Set the working directory
WORKDIR /workspace

# Copy your script into the container
COPY inference.py /workspace

# Copy downloaded model into container
COPY gpt2_model /workspace/gpt2_model

# Copy downloaded tokenizer into container
COPY gpt2_tokenizer /workspace/gpt2_tokenizer

# Specify the command to run
CMD ["bin/python", "inference.py"]
