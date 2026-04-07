# qwen_test_triples_generate.py
from dataset.hico_dataset import HICODataset
import torch
import torch.nn as nn
import os
import glob  
import json
from tqdm import tqdm
import shutil
import dashscope
from http import HTTPStatus
import pickle
import time
from qwen_long import extract_triples

# ==================== Configure parameters ====================
TEST_PROMPTS_FILE = "../test_prompts.json"
TRIPLES_TEST_FILE = "../test_triples_update2.json" # a new test dataset
HICO_PKL_PATH = "../DATA/hico_det_test.pkl"
UPDATED_HICO_PKL = "../DATA/hico_det_test_with_triples_update2.pkl"

MAX_RETRIES = 3
BATCH_SIZE = 4

# Qwen API
QWEN_API_KEY = "sk-.."  
QWEN_MODEL = "qwen-long"

# ==================== Qwen API ====================
dashscope.api_key = QWEN_API_KEY
# ==================== 1. Generate additional triples ====================
def process_test_prompts():
    """Processing test_prompts.json generates triples"""
    with open(TEST_PROMPTS_FILE, 'r') as f:
        test_data = json.load(f)
    
    if os.path.exists(TRIPLES_TEST_FILE):
        os.remove(TRIPLES_TEST_FILE)
    
    results = []
    for i in tqdm(range(0, len(test_data), BATCH_SIZE), desc="Processing Test Prompts"):
        batch = test_data[i:i+BATCH_SIZE]
        for item in batch:
            caption = item.get("prompt", "").strip()
            if not caption:
                continue
            
            triples = extract_triples(caption, version="default")
            
            print(f"File Name: {item['file_name']}")
            print(f"Prompt: {caption}")
            print(f"Triples: {triples}")

            results.append({
                "file_name": item["file_name"],
                "prompt": caption,
                "triples": triples
            })
    
    with open(TRIPLES_TEST_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Test triples saved to {TRIPLES_TEST_FILE}")

if __name__ == "__main__":
    # Executive-Function 1: Generate triples
    if not os.path.exists(TRIPLES_TEST_FILE):
        process_test_prompts()
    else:
        print(f"{TRIPLES_TEST_FILE} already exists, skipping processing")
    
    # Executive-Function 2: Update the .pkl file
    # if os.path.exists(TRIPLES_TEST_FILE):
    #     update_hico_pkl()
    # else:
    #     print(f"{TRIPLES_TEST_FILE} not found, cannot update HICO data")

