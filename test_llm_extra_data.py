
# ==================== Generate additional triplets of inference data and update hico_det_test.pkl ====================
from dataset.hico_dataset import HICODataset
import torch
import torch.nn as nn
import os
import glob  
import json
from tqdm import tqdm
import shutil
from llm import extract_triples
import pickle

TEST_PROMPTS_FILE = "../test_prompts.json"
TRIPLES_TEST_FILE = "../test_triples.json" # a new test dataset
HICO_PKL_PATH = "../DATA/hico_det_test.pkl"
UPDATED_HICO_PKL = "../DATA/hico_det_test_with_triples.pkl"

MAX_RETRIES = 3
BATCH_SIZE = 4
LLAMA_MODEL_PATH = "Llama-2-7b-chat-hf"

# def init_components():
    
#     with open('../template.txt', 'r') as f:
#         global PROMPT_TEMPLATE
#         PROMPT_TEMPLATE = f.read()

# # ==================== 1. Generate additional triples ====================
def process_test_prompts():
    with open(TEST_PROMPTS_FILE, 'r') as f:
        test_data = json.load(f)  # Expected structure：[{"file_name": "xx.jpg", "prompt": "..."}, ...]
    
    # test_data = test_data[:12]   

    if os.path.exists(TRIPLES_TEST_FILE):
        os.remove(TRIPLES_TEST_FILE) 

    results = []
    for i in tqdm(range(0, len(test_data), BATCH_SIZE), desc="Processing Test Prompts"):
        batch = test_data[i:i+BATCH_SIZE]
        for item in batch:
            for attempt in range(MAX_RETRIES):
                try:
                    triples = extract_triples(
                        prompt=item["prompt"],
                        version="default",
                        model_path=LLAMA_MODEL_PATH
                    )
                    results.append({
                        "file_name": item["file_name"],
                        "prompt": item["prompt"],
                        "triples": triples
                    })
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES-1:
                        results.append({
                            "file_name": item["file_name"],
                            "error": str(e)
                        })

    with open(TRIPLES_TEST_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Test triples saved to {TRIPLES_TEST_FILE}")

# ==================== Review the fields for each piece of data in hico_det_test.pkl ====================

PKL_PATH = "../DATA/hico_det_test.pkl"

with open(PKL_PATH, 'rb') as f:
    data = pickle.load(f)


for i, item in enumerate(data[:3]):
    print(item)


# ==================== 2. Update hico_det_test.pkl====================
# def update_hico_pkl():
#     with open(HICO_PKL_PATH, 'rb') as f:
#         hico_data = pickle.load(f)
    
#     with open(TRIPLES_TEST_FILE, 'r') as f:
#         triples_data = json.load(f)
    
#     triples_map = {item["file_name"]: item for item in triples_data if "triples" in item}
    
#     updated_count = 0
#     for item in tqdm(hico_data, desc="Updating HICO Data"):
#         file_name = item["file_name"]
#         if file_name in triples_map:
#             item["triples"] = triples_map[file_name]["triples"]
#             updated_count += 1
    
#     with open(HICO_PKL_PATH, 'wb') as f:
#         pickle.dump(hico_data, f)
#     print(f"Updated {updated_count}/{len(hico_data)} items. Saved to {HICO_PKL_PATH}")

# if __name__ == "__main__":
#     # init_components()


#     if not os.path.exists(TRIPLES_TEST_FILE):
#          process_test_prompts()
#     else:
#          print(f"{TRIPLES_TEST_FILE} already exists, skipping processing")
    
#     if os.path.exists(TRIPLES_TEST_FILE):
#         update_hico_pkl()
#     else:
#         print(f"{TRIPLES_TEST_FILE} not found, cannot update HICO data")

# with open(TRIPLES_TEST_FILE, 'r') as f:
#     sample = json.load(f)[0]
#     print(sample.keys()) 

# with open(UPDATED_HICO_PKL, 'rb') as f:
#     sample_data = pickle.load(f)[0]
#     print(sample_data.get('triples', 'No triples field'))

# with open(UPDATED_HICO_PKL, 'rb') as f:
#     data = pickle.load(f) 
    
#     for idx, item in enumerate(data[:6], 1):
#         print(json.dumps(item, indent=2, ensure_ascii=False, default=str)) 
