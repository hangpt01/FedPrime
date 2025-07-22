import torch
from torch import nn
import torchvision.models as models
import math
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import os
from utils.fmodule import FModule

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class Pool(FModule):
    def __init__(self, patch_size=32, embed_dim=768, pool_size=20, top_k=5, dropout_value=0.0):
        super(Pool, self).__init__()
        patch_size_pair = _pair((patch_size, patch_size))
        self.top_k = top_k
        self.pool_size = pool_size
        self.prompt = nn.Parameter(torch.zeros(pool_size, embed_dim))
        self.features_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.features_dropout = nn.Dropout(dropout_value)

        # Prompt initialization (uniform distribution)
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size_pair, 1) + embed_dim))
        nn.init.uniform_(self.prompt.data, -val, val)
        nn.init.uniform_(self.features_proj.weight, -1, 1)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, cls_features=None):
        current_pool_size = self.prompt.shape[0]
        
        if cls_features is None:
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        else:
            x_embed_mean = cls_features
            
        prompt_norm = self.l2_normalize(self.prompt, dim=1) # Pool_size, C
        x_embed_mean = self.features_proj(self.features_dropout(x_embed_mean))
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C
        similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
        _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
        # if self.batchwise_prompt:
        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
        if prompt_id.shape[0] < current_pool_size:
            prompt_id = torch.cat([prompt_id, torch.full((current_pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
            id_counts = torch.cat([id_counts, torch.full((current_pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
        _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
        major_prompt_id = prompt_id[major_idx] # top_k
        # expand to batch
        idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
        self.top_k_idx = idx[0]
        
        # print("Top k", self.top_k_idx)
        batched_prompt = self.prompt[idx] # B, top_k, C
        batched_key_norm = prompt_norm[idx]
        
        x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * x_embed_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed.shape[0]
        
        return reduce_sim, batched_prompt

class Pooler(FModule):
    def __init__(self):
        super().__init__()
        hidden_size = 768
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class Classifier(FModule):
    def __init__(self):
        super().__init__()
        self.hidden_size = 768
        cls_num = 8
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.LayerNorm(self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, cls_num),
        )
    
    def forward(self, x):
        return self.classifier(x)
    

class Model(FModule):
    def __init__(self, pool_size=10, top_k=5, num_classes=10):
        super(Model, self).__init__()

        self.device = None
        self.transformer = None
        self.text_embeddings = None
        # import pdb; pdb.set_trace()  
        
        self.hparams_config = {'hidden_size': 768,
                               'pool_size': 20,
                               'top_k':10,
                                'max_image_len': 40}

        self.token_type_embeddings = nn.Embedding(2, self.hparams_config["hidden_size"])
        self.token_type_embeddings.apply(init_weights)
        
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=False

        # Pool and Pooler
        self.pool = Pool(embed_dim=768, pool_size=self.hparams_config["pool_size"], top_k=self.hparams_config['top_k'])
        self.pooler = Pooler()
        self.pooler.apply(init_weights)
        
        # Classifier
        self.classifier = Classifier()
        self.classifier.apply(init_weights)   
        
        # prompt
        embed_dim = self.hparams_config["hidden_size"]
        
        self.trained_prompts_checklist = torch.zeros(self.pool.prompt.shape[0], dtype=torch.float32)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.prompts = None
        
        
    def infer(self, batch, transformer, text_embeddings):
        self.transformer = transformer
        self.text_embeddings = text_embeddings

        imgkey = "image"

        do_mlm = ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        # import pdb; pdb.set_trace()
        text_embeds = self.text_embeddings(text_ids)
        img = batch[imgkey][0]     
        self.device = img.device

        # import pdb; pdb.set_trace()
        (
            image_embeds,
            image_masks,
            patch_index,
            image_labels,
        ) = self.transformer.visual_embed(
            img,
            max_image_len=self.hparams_config["max_image_len"],
            mask_it=False,
        )

        # import pdb; pdb.set_trace()
        # (batch, 40, 768), (batch, 217, 768)
        text_embeds, image_embeds = (        
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)), # + (batch,40)
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, 1)                  # + (batch,217)
            ),
        )
        
        # co_masks = torch.cat([text_masks, image_masks], dim=1)    # torch.Size([batch, 329]);     batch, 353=257+96
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)       # torch.Size([1, 233, 768])             batch, 257, 768
        x = co_embeds.detach()      # torch.Size([1, 233, 768])     batch, 257, 768=text+img

        n = x.shape[0]    
        reduce_sim, batched_prompt = self.pool(x, cls_features=None)
        # self.prompts = batched_prompt       # B, topk, 768
        # import pdb; pdb.set_trace()
        prompt_masks = torch.ones(batched_prompt.shape[0], batched_prompt.shape[1], dtype=batched_prompt.dtype, device=batched_prompt.device).long()
        co_masks = torch.cat([prompt_masks, text_masks, image_masks], dim=1)    # torch.Size([batch, 329]);     batch, 353=257+96
        # print(self.pool.pool_size, self.pool.top_k, batched_prompt.shape)
        x = torch.cat((batched_prompt, x), dim=1)

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)
     

        x = self.transformer.norm(x)
        cls_feats = self.pooler(x)
        return cls_feats, batched_prompt


    def checking_trained_prompt(self):
        if self.pool.top_k_idx.device != self.trained_prompts_checklist.device:
            self.trained_prompts_checklist = self.trained_prompts_checklist.to(self.pool.top_k_idx.device)
        self.trained_prompts_checklist[self.pool.top_k_idx] += 1.0
        

    def forward(self, transformer, text_embeddings, batch, flag="", client_id=None, current_round=None):
        
        cls_feats, batched_prompt = self.infer(batch, transformer, text_embeddings)

        # Capture embeddings before the classifier
        embedding_before_classifier = cls_feats.detach().cpu()

        # Classify and get embeddings after the classifier
        imgcls_logits = self.classifier(cls_feats)
        embedding_after_classifier = imgcls_logits.detach().cpu()

        # Save to a dictionary
        sample_data = {
            "local_prompts": batched_prompt.detach().cpu(),
            "missing_type": batch["missing_type"],
            "embedding_before_classifier": embedding_before_classifier,
            "embedding_after_classifier": embedding_after_classifier,
        }
        save_file = False
        if save_file:
            if flag != "":
                if flag=="train" and current_round % 25 == 0:
                    os.makedirs(f"output/imdb/L2P/train/client_{client_id+1}/", exist_ok=True)
                    file_path = f"output/imdb/L2P/train/client_{client_id+1}/sample_data_round_{current_round}.pt"
                    if os.path.exists(file_path):
                    # Load existing data
                        existing_data = torch.load(file_path)
                        if isinstance(existing_data, list):
                            existing_data.append(sample_data)
                        else:
                            existing_data = [existing_data, sample_data]
                        # Save the updated list
                        torch.save(existing_data, file_path)
                    else:
                        # Save a new list with the current sample data
                        torch.save([sample_data], file_path)
                        # Append to file if it exists, else create new
                elif flag=="test" and current_round % 25 == 0:
                    os.makedirs("output/imdb/L2P/test/", exist_ok=True)
                    file_path = f"output/imdb/L2P/test/sample_data_round_{current_round}.pt"
                    if os.path.exists(file_path):
                        # Load existing data
                        existing_data = torch.load(file_path)
                        if isinstance(existing_data, list):
                            existing_data.append(sample_data)
                        else:
                            existing_data = [existing_data, sample_data]
                        # Save the updated list
                        torch.save(existing_data, file_path)
                    else:
                        # Save a new list with the current sample data
                        torch.save([sample_data], file_path)
            

        imgcls_labels = batch["label"]

        imgcls_labels = torch.tensor(imgcls_labels).to(self.device).long()

        imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)



        return imgcls_loss, imgcls_loss, imgcls_logits
