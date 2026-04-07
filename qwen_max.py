import torch
import torch.nn as nn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import transformers
from transformers import pipeline
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import CLIPTextModel, AutoTokenizer
import re
import requests
import json
import os
import torch.nn.functional as F
import dashscope
from http import HTTPStatus

from openai import OpenAI
import os

#- - - - - Extract triples- - -method - - #
def extract_triples(prompt, version, model_path=None):
    os.environ["DASHSCOPE_API_KEY"] = "sk.." 
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", 
    )

    with open('../template.txt', 'r') as f:
        template = f.read().strip()
    user_textprompt = f"Caption:{prompt} \n Let's start thinking about it:" 
    textprompt = f"{' '.join(template)} \n {user_textprompt}"

    response = client.chat.completions.create(
        model="qwen-max-latest",
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': textprompt}
        ],
        extra_body={"enable_thinking": False}
    )

    output = response.choices[0].message.content

    triples_dict = output
    triples = []
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, triples_dict)

    for match in matches:
        h = match[0].strip().lower()
        r = match[1].strip().lower()
        t = match[2].strip().lower()
        if h != 'h' and r != 'r' and t != 't': 
            triples.append((h, r, t))

    return triples # obtain implicit interactive relationships