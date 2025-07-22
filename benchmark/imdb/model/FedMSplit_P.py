from typing import Any
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from utils.fmodule import FModule


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


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()
    

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
    def __init__(self):
        super(Model, self).__init__()
        self.hparams_config = {'prompt_type': 'input', 
                                'prompt_length': 16, 
                                'learnt_p': True, 
                                'prompt_layers': [0, 1, 2, 3, 4, 5], 
                                'multi_layer_prompt': True, 
                                'vocab_size': 30522, 
                                'hidden_size': 768, 
                                'num_heads': 12, 
                                'num_layers': 12, 
                                'drop_rate': 0.1,
                                'max_image_len': 40}
        
        self.device = None
        self.transformer = None
        self.text_embeddings = None
        # import pdb; pdb.set_trace()  

        self.token_type_embeddings = nn.Embedding(2, self.hparams_config["hidden_size"])
        self.token_type_embeddings.apply(init_weights)
        
        for param in self.token_type_embeddings.parameters():
            param.requires_grad=False

        self.pooler = Pooler()
        self.pooler.apply(init_weights)

        self.classifier = Classifier()
        self.classifier.apply(init_weights)   
  
        # prompt
        self.prompt_type = self.hparams_config["prompt_type"]
        prompt_length = self.hparams_config["prompt_length"]
        self.prompt_length = prompt_length
        embed_dim = self.hparams_config["hidden_size"]
        self.learnt_p = self.hparams_config["learnt_p"]
        self.prompt_layers = self.hparams_config["prompt_layers"]
        self.multi_layer_prompt = self.hparams_config["multi_layer_prompt"]
        prompt_num = len(self.prompt_layers) if self.multi_layer_prompt else 1

        complete_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        complete_prompt[:,0:1,:].fill_(1) 
       
        # import pdb; pdb.set_trace()

        # self.
        self.complete_prompt = nn.Parameter(complete_prompt)                    # [6, 16, 768])

        missing_text_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_text_prompt[:,2:3,:].fill_(1)            
        self.missing_text_prompt = nn.Parameter(missing_text_prompt)            # [6, 16, 768])

        missing_img_prompt = torch.zeros(prompt_num, prompt_length, embed_dim)
        missing_img_prompt[:,1:2,:].fill_(1)            
        self.missing_img_prompt = nn.Parameter(missing_img_prompt)              # [6, 16, 768])

        
    def infer(
            self,
            batch,
            transformer,
            text_embeddings,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            image_embeds=None,
            image_masks=None,
        ):
        self.transformer = transformer
        self.text_embeddings = text_embeddings
        # import pdb; pdb.set_trace()
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        # import pdb; pdb.set_trace()
        text_embeds = self.text_embeddings(text_ids)
        img = batch[imgkey][0]     
        self.device = img.device

        # import pdb; pdb.set_trace()
        if image_embeds is None and image_masks is None:
                   
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=self.hparams_config["max_image_len"],
                mask_it=mask_image,
            )
    

        else:
            patch_index, image_labels = (
                None,
                None,
            )
        # import pdb; pdb.set_trace()
        # (batch, 40, 768), (batch, 217, 768)
        text_embeds, image_embeds = (        
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)), # + (batch,40)
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)                  # + (batch,217)
            ),
        )
        
        # instance wise missing aware prompts
        prompts = None
        for idx in range(len(img)):
            if batch["missing_type"][idx] == 0:
                prompt = self.complete_prompt        
            elif batch["missing_type"][idx] == 1:
                # import pdb; pdb.set_trace()
                prompt = self.missing_text_prompt
            elif batch["missing_type"][idx] == 2:
                prompt = self.missing_img_prompt
                # 3 prompt: ([6, 16, 768])
            if prompt.size(0) != 1:
                prompt = prompt.unsqueeze(0)
            
            if prompts is None:
                prompts = prompt
            else:
                prompts = torch.cat([prompts, prompt], dim=0)      #  ([batch, 6, 16, 768])
        # import pdb; pdb.set_trace()     
        if self.learnt_p:
            if self.prompt_type=='attention':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length//2, dtype=prompts.dtype, device=prompts.device).long()
            elif self.prompt_type=='input':
                prompt_masks = torch.ones(prompts.shape[0], self.prompt_length*len(self.prompt_layers), dtype=prompts.dtype, device=prompts.device).long()      #torch.Size([batch, 96])
                # import pdb; pdb.set_trace()
        else:
            prompt_masks = torch.ones(prompts.shape[0], self.prompt_length, dtype=prompts.dtype, device=prompts.device).long()   
        
        co_masks = torch.cat([prompt_masks, text_masks, image_masks], dim=1)    # torch.Size([batch, 329]);     batch, 353=257+96
        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)       # torch.Size([1, 233, 768])             batch, 257, 768
        # import pdb; pdb.set_trace()
        x = co_embeds.detach()      # torch.Size([1, 233, 768])     batch, 257, 768=text+img

        for i, blk in enumerate(self.transformer.blocks):
            if i in self.prompt_layers:
                if self.multi_layer_prompt:
                    x, _attn = blk(x, mask=co_masks, 
                                   prompts=prompts[:,self.prompt_layers.index(i)],      # batch, 16, 768
                                   learnt_p=self.learnt_p,
                                   prompt_type=self.prompt_type)
                else:
                    x, _attn = blk(x, mask=co_masks, prompts=prompts, learnt_p=self.learnt_p)
            else:
                x, _attn = blk(x, mask=co_masks)
        # import pdb; pdb.set_trace()
        # x: torch.Size([1, 329, 768])
        x = self.transformer.norm(x)    # x: torch.Size([1, 329, 768])
        
        
        if self.prompt_type == 'input':
            total_prompt_len = len(self.prompt_layers)* prompts.shape[-2]   # len([0, 1, 2, 3, 4, 5]) * 16
        elif self.prompt_type == 'attention':
            total_prompt_len = prompts.shape[-2]
        
        text_feats, image_feats = (
            x[:,total_prompt_len : total_prompt_len+text_embeds.shape[1]],
            x[:, total_prompt_len+text_embeds.shape[1] :],
        )       # ([1, 40, 768]), ([1, 193, 768])
        if self.prompt_type == 'input':
            cls_feats = self.pooler(x[:,total_prompt_len:total_prompt_len+1])   
#         cls_feats = self.pooler(x[:,:prompts.size(1)].mean(dim=1,keepdim=True))
        elif self.prompt_type == 'attention':
            cls_feats = self.pooler(x)
            
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret["cls_feats"], prompts


    def forward(self, transformer, text_embeddings, batch, flag="", client_id=None, current_round=None):
        
        cls_feats, prompts = self.infer(batch, transformer, text_embeddings)

        # Capture embeddings before the classifier
        embedding_before_classifier = cls_feats.detach().cpu()

        # Classify and get embeddings after the classifier
        imgcls_logits = self.classifier(cls_feats)
        embedding_after_classifier = imgcls_logits.detach().cpu()

        dataset = "imdb"
        model = "fedmsplit"     

        # Save to a dictionary
        sample_data = {
            "prompts": prompts.detach().cpu(),
            "missing_type": batch["missing_type"],
            "label": batch["label"],
            "embedding_before_classifier": embedding_before_classifier,
            "embedding_after_classifier": embedding_after_classifier,
        }
        save_file = False
        if save_file:
            if flag != "":
                if flag == "train":
                    if current_round == 1 or current_round % 25 == 0:
                        # Create the folder for saving client-specific files
                        output_dir = f"output/{dataset}/{model}/train/client_{client_id+1}/"
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Loop over each key in sample_data and save individually
                        for key, value in sample_data.items():
                            file_path = f"{output_dir}{key}_round_{current_round}.pt"
                            torch.save(value, file_path)
                
                elif flag == "test":
                    if current_round == 2 or current_round % 25 == 0:
                        # Create the folder for test files
                        output_dir = f"output/{dataset}/{model}/test/"
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Save each key individually in the test directory
                        for key, value in sample_data.items():
                            file_path = f"{output_dir}{key}_round_{current_round}.pt"
                            torch.save(value, file_path)
            

        imgcls_labels = batch["label"]

        imgcls_labels = torch.tensor(imgcls_labels).to(self.device).long()

        imgcls_loss = F.cross_entropy(imgcls_logits, imgcls_labels)

        return imgcls_loss, imgcls_loss, imgcls_logits
