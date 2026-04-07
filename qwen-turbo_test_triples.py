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

#- - - - - Extract triples- - -method - - #
def extract_triples(prompt, version, model_path=None):
    dashscope.api_key = "sk.." 
    QWEN_MODEL = "qwen-turbo"

    # Read the template and build the prompt
    with open('../template.txt', 'r') as f:
        template = f.readlines()
    user_textprompt = f"Caption:{prompt} \n Let's start thinking about it:" 
    textprompt = f"{' '.join(template)} \n {user_textprompt}"

    # Call the Qwen API
    response = dashscope.Generation.call(
        model=QWEN_MODEL,
        prompt=textprompt
    )

    if response.status_code == HTTPStatus.OK:
        output = response.output.text
    else:
        raise Exception(f"Qwen API Error: {response.code} - {response.message}")

    # Parse extracted triplet dictionary
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

# def get_params_dict(output_text):
    
#     response = output_text  
#     # Find Output
#     output_match = re.search(r"Output: (.*?)(?=\n\n|\Z)", response, re.DOTALL) 
#     if output_match:
#         out = output_match.group(1).strip()  
#         #print("Output:", output)
#     else:
#         out = []
#         print("Output not found.")

#     dict = {'Output': out}
#     return dict