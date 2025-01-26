# GENAI-POC

## Description
POC about GENAI

## Installation
Steps to install your project:
```sh
# create venv
conda create -p genaivenv python=3.11
conda create -n genaivenv python=3.12

# Powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# activate the env
conda activate D:\\Projects\\genai-poc\\genaivenv

```

## Ollama Docker Setup on local
```sh
# Pull Image 
docker pull ollama/ollama

# Run Ollama Image
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Execute ollama container
docker exec -it ollama ollama run llama3
```

## Redis Docker Setup on local
```sh
docker pull redis

docker run -it --network redis-network --rm redis redis-cli -h redis-docker

docker run -d --name redis -d redis redis-server --save 60 1 --loglevel warning -p 6379:6379 redis --bind 0.0.0.0

docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest


```

## NLTK
```sh

import nltk
nltk.download('punkt_tab')

```