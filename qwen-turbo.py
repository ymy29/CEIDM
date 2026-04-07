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

#- - - - - Extract triples- - -method - - #
def extract_triples(prompt, version, model_path=None):
    if model_path == None:
        model_id = "Llama-2-7b-chat-hf"
    else:
        model_id = model_path

    tokenizer = LlamaTokenizer.from_pretrained(model_id) 
    model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map='auto',
                                             torch_dtype=torch.float16) 

    with open('../template.txt', 'r') as f:
        template = f.readlines() 

        user_textprompt = f"Caption:{prompt} \n Let's start thinking about it:" 
        textprompt = f"{' '.join(template)} \n {user_textprompt}"  
        model_input = tokenizer(textprompt, return_tensors="pt").to("cuda")  
        with torch.no_grad():  
            res = model.generate(**model_input, max_new_tokens=1024)[0]  
            output = tokenizer.decode(res, skip_special_tokens=True)  
            output = output.replace(textprompt, '') 
            if "Output: " not in output:
                output = "Output: " + output.lstrip() 


    # Parsing extracted triplet dictionary
    triples_dict = get_params_dict(output)
    triples = []
    pattern = r'\(([^,]+),\s*([^,]+),\s*([^)]+)\)'
    matches = re.findall(pattern, triples_dict['Output'])

    for match in matches:
        h = match[0].strip().lower()
        r = match[1].strip().lower()
        t = match[2].strip().lower()
        if h != 'h' and r != 'r' and t != 't': 
            triples.append((h, r, t))

    return triples  # obtain implicit interactive relationships

def get_params_dict(output_text):
    
    response = output_text  
    # Find Output
    output_match = re.search(r"Output: (.*?)(?=\n\n|\Z)", response, re.DOTALL)  
    if output_match:
        out = output_match.group(1).strip()  
    else:
        out = []
        print("Output not found.")

    dict = {'Output': out}
    return dict