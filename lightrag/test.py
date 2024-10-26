import requests
from lightrag import LightRAG, QueryParam
from lightrag.llm import hf_embedding, openai_complete_if_cache
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer
from lightrag.prompt import *
from lightrag.operate import *
import os
from utils import load_json
WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await llm_service(
        "qwen2.5",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )


async def llm_service(model_name, prompt, system_prompt, history_messages, **kwargs):
    url = r"http://10.132.90.4:8090/v1/chat/completions"
    data = {
        "model": "/data/llm_model/pretrained_model/qwen/Qwen2.5-72B-Instruct-AWQ",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "repetition_penalty": 1.0,
        "top_p": 0.1,
        "max_tokens": 2048,
        "stream": "False"
    }
    res = requests.post(url=url, json=data)
    return res.json()["choices"][0]["message"]["content"]


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: hf_embedding(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(r"D:\Users\057568\Desktop\bge-m3"),
            embed_model=AutoModel.from_pretrained(r"D:\Users\057568\Desktop\bge-m3")
        )
    )
)

if __name__ == "__main__":
    json_data = load_json(file_path=r"./data/中国视神经脊髓炎谱系疾病诊断与治疗指南.json")
    for chunk in json_data:
        text = chunk["text"]
        rag.insert(text)
