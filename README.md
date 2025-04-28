# GENAI-POC

## Description
POC about GENAI

**References**

https://platform.openai.com/api-keys
https://platform.openai.com/docs/models/overview

https://python.langchain.com/docs/get_started/introduction
https://python.langchain.com/docs/get_started/quickstart


**Setup Commands**
```commandline
pip install langchain
pip install openai
pip install huggingface_hub transformers accelerate bitsandbytes
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
huggingface-cli login
```

https://stackoverflow.com/questions/48152674/how-do-i-check-if-pytorch-is-using-the-gpu



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


## LangChain

1. Ollama
    https://python.langchain.com/api_reference/ollama/llms/langchain_ollama.llms.OllamaLLM.html
    https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html#langchain_ollama.chat_models.ChatOllama

2. Open AI