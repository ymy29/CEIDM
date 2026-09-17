import torch
import torch.nn as nn
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import CLIPTextModel, AutoTokenizer
import os
import torch.nn.functional as F
from inspect import isfunction
import math
from einops import rearrange, repeat
import json
# from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
from torch.utils import checkpoint

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("Xformers is not available. Install via ")

class TestEmbedding(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()

        # Load the CLIP text encoder
        self.clip_model = CLIPTextModel.from_pretrained("../clip-vit-base-patch32")
        self.clip_tokenizer = AutoTokenizer.from_pretrained("../clip-vit-base-patch32")

        # Independent linear transform layer
        self.f_h = nn.Linear(embed_dim, embed_dim) 
        self.f_r = nn.Linear(embed_dim, embed_dim) 
        self.f_t = nn.Linear(embed_dim, embed_dim) 

        self.norm_h = nn.LayerNorm(embed_dim)
        self.norm_r = nn.LayerNorm(embed_dim)
        self.norm_t = nn.LayerNorm(embed_dim)

        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads=4)

        self.dim_adapter = nn.Linear(embed_dim, 768)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def clip_text_encoder(self, text):
        """
        Use the CLIP text encoder to generate embedding vectors
        :param text: Enter text (str or List[str])
        :return: CLIP text embedding vector
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.to(device)
        inputs = self.clip_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1) 

    def forward(self, batch):
        """
        Encode the triple for each sample in the batch
        :param batch_triples: List[List[Tuple[str, str, str]]], each sample contains multiple triples (h, r, t)
        :return: torch. Tensor, shaped like (B, N_triples*3, 768)
        """
        json_file_path = "../test_triples_update.json"
        with open(json_file_path, 'r') as f:
            triples_data = json.load(f)

        caption_to_triples = {tuple(item['prompt']): item['triples'] for item in triples_data}

        batch_triples = []
        final_embeddings = []
        if isinstance(batch["prompt"], str):
            batch["prompt"] = [batch["prompt"]]

        for prompt in batch["prompt"]:
            prompt_tuple = tuple(prompt)  
            if prompt_tuple == ():
                final_embeddings = torch.zeros(1, 90, 512)
            else:
                Triples = caption_to_triples.get(prompt_tuple, [])
                batch_triples.append(Triples)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"batch_triples: {batch_triples} ") 
        final_embeddings = []
        for triples1 in batch_triples:
            if not triples1:  # triples1 = null
                triples1_embeddings = []
                triples1_embeddings_padded = []
                triples1_embeddings = torch.empty((0, 512)).to(device) # (0,512)
                #print(f"triples1_embeddings: {triples1_embeddings} ") 
                triples1_embeddings = triples1_embeddings.unsqueeze(0)
                #print(f"triples1_embeddings.shape: {triples1_embeddings.shape}")
                
                target_size = 90
                triples1_embeddings_padded = F.pad(triples1_embeddings, (0, 0, 0, target_size))
                final_embeddings.append(triples1_embeddings_padded)
            else:
                sample_embeddings = []
                triples1_embeddings = []
                triples1_embeddings_padded = []
                for (h, r, t) in triples1:

                    h_emb = self.clip_text_encoder(h)
                    r_emb = self.clip_text_encoder(r)
                    t_emb = self.clip_text_encoder(t)

                    h_emb = self.f_h(h_emb)
                    r_emb = self.f_r(r_emb)
                    t_emb = self.f_t(t_emb)

                    combined = torch.stack([h_emb, r_emb, t_emb], dim=0)
                    resi, _ = self.self_attention(combined, combined, combined)  
                    #resi = self.transformer_layer(combined)  

                    # Improvement
                    # attn_output, _ = self.self_attention(combined, combined, combined)
                    # attn_output = self.norm1(combined + attn_output) 
                    
                    # ffn_output = self.ffn(attn_output)
                    # ffn_output = self.norm2(ffn_output)
                    # resi = self.norm2(attn_output + ffn_output) 

                    # Aggregate output from the attention layer
                    resi = resi.mean(dim=0) 

                    # Xh = h_emb + resi  
                    # Xr = r_emb + resi  
                    # Xt = t_emb + resi  

                    Xh = self.norm_h(h_emb + resi)
                    Xr = self.norm_r(r_emb + resi)
                    Xt = self.norm_t(t_emb + resi)

                    sample_embeddings.append((Xh, Xr, Xt))
                triples1_embeddings = torch.cat([torch.cat([Xh, Xr, Xt], dim=0) for Xh, Xr, Xt in sample_embeddings], dim=0).to(device) #[N*3,512], N represents the number of triples in a sample
                #print(f"triples1_embeddings: {triples1_embeddings} ") 
                triples1_embeddings = triples1_embeddings.unsqueeze(0)
                #print(f"triples1_embeddings.shape: {triples1_embeddings.shape}")

                target_size = 90
                current_size = triples1_embeddings.size(1)
                if current_size > target_size:
                    # If the current size exceeds the target size, truncation is performed
                    triples1_embeddings_padded = triples1_embeddings.narrow(1, 0, target_size)
                else:
                    padding_size = target_size - current_size
                    # Use F.pad for padding
                    triples1_embeddings_padded = F.pad(triples1_embeddings, (0, 0, 0, padding_size))

                final_embeddings.append(triples1_embeddings_padded)
        final_embeddings = torch.cat(final_embeddings, dim=0).to(device) # [B, N*3, 512]
        kgs = self.dim_adapter(final_embeddings)
        return kgs

# kgs is implicit interactive information.