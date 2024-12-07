# gpu-container-example

An example of running LLMs in GPU-accelerated containerized environments.

## Prerequisites

- A machine with a CUDA-compatible GPU (e.g., NVIDIA GPUs)
- Docker
- NVIDIA Container Toolkit

## Steps

Download the model:
```bash
python3 download.py
```

Build the container:
```bash
docker build -t llm-gpu .
```

Run the container:
```bash
docker run --gpus all llm-gpu
```
