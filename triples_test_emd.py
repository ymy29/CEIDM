from dataset.hico_dataset import HICODataset
import torch
import torch.nn as nn
import os
import glob  
import json
from tqdm import tqdm
import shutil
from llm import extract_triples

DATA_DIR = "../FGAHOI/data/hico_20160224_det/hico_det_clip"
CAPTION_FILE = "../captions.txt"
TRIPLES_FILE = "../triples_results.json" # a new tain dataset
NEW_DATA_DIR = "../dataset_with_triples"
MAX_RETRIES = 3
BATCH_SIZE = 4
LLAMA_MODEL_PATH = "Llama-2-7b-chat-hf"

def init_components():
    os.makedirs(NEW_DATA_DIR, exist_ok=True)
    with open('../template.txt', 'r') as f:
        global PROMPT_TEMPLATE
        PROMPT_TEMPLATE = f.read()
def process_captions():

    with open(CAPTION_FILE, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    caption_data = []
    for line in lines:
        if not line:
            continue
        try:
            file_part = line.split("File: ")[1].split(", Caption: ")
            file_path = file_part[0].strip()
            caption = file_part[1].strip()
            caption_data.append((file_path, caption))
        except:
            print(f"Failed to parse line: {line}")

    if os.path.exists(TRIPLES_FILE):
        os.remove(TRIPLES_FILE) 

    results = []
    for i in tqdm(range(0, len(caption_data), BATCH_SIZE), desc="Processing Captions"):
        batch = caption_data[i:i+BATCH_SIZE]
        for file_path, caption in batch:
            for attempt in range(MAX_RETRIES):
                try:
                    triples = extract_triples(
                        prompt=caption,
                        version="default",
                        model_path=LLAMA_MODEL_PATH
                    )
                    results.append({
                        "file_path": file_path,
                        "caption": caption,
                        "triples": triples
                    })
                    break
                except Exception as e:
                    if attempt == MAX_RETRIES-1:
                        results.append({
                            "file_path": file_path,
                            "error": str(e)
                        })

    with open(TRIPLES_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Triples saved to {TRIPLES_FILE}")

def update_data_files():
    """Add triples to the original data file as new meta information"""
    with open(TRIPLES_FILE, 'r') as f:
        triples_data = json.load(f)
    
    triples_map = {item["file_path"]: item for item in triples_data} 

    for file_path in tqdm(glob.glob(os.path.join(DATA_DIR, "*.pt")), desc="Updating Files"):
        try:
            data = torch.load(file_path, map_location="cpu")
   
            #file_key = os.path.basename(file_path)
            triples_info = triples_map.get(file_path, {})
            
            if "triples" in triples_info:
                data["triples"] = triples_info["triples"]
                
                torch.save(data, file_path)  

            else:
                print(f"No triples found for {file_path}")
                
        except Exception as e:
            print(f"Failed to process {file_path}: {str(e)}")

if __name__ == "__main__":
    
    init_components()
    if not os.path.exists(TRIPLES_FILE):
        process_captions()
    else:
        print(f"{TRIPLES_FILE} already exists, skipping processing")

    update_data_files()
    print("Data update completed. New data saved to:", NEW_DATA_DIR)
