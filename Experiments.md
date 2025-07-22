# Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data

## Training Details and Hyperparameters

This repository implements the federated prompt-tuning framework described in the ICCV 2025 paper.

### Datasets
- UPMC Food-101: 6,728 multimodal samples
- MM-IMDB: 5,778 image-text pairs after preprocessing

### Input Preprocessing
- Text inputs are tokenized with BERT-base-uncased tokenizer  
  - Max sequence length: 40 (Food-101), 128 (MM-IMDB)  
  - Missing text inputs replaced with empty strings  
- Images resized with shorter side = 384 pixels, longer side ≤ 640 pixels  
- Images decomposed into 32×32 patches  
- Missing images replaced with dummy images filled with ones

### Model and Prompt Tuning
- Backbone: Pretrained ViLT transformer (frozen during training)  
- Trainable components: learnable prompt pools and task-specific layers (pooler and classifier)  
- Prompt pools contain 20 prompts each  
- Prompt dimension: 768 (matching ViLT embedding dimension)  
- For FED-PRIME, 5 prompts are selected from intra-client and 5 from inter-client pools per input, concatenated and added to the first Multi-Head Self-Attention layer  
- Prompt clusters are adaptively learned during the optimization process via a clustering-based alignment mechanism  

### Federated Training Setup
- Number of clients: 20  
- All clients participate in every communication round  
- Total communication rounds: 250  
- Each client performs 1 epoch of local training per round

### Optimization
- Optimizer: Stochastic Gradient Descent (SGD)  
- Learning rate: 0.05 (Food-101), 0.01 (MM-IMDB)  
- Batch size: 512 (Food-101), 256 (MM-IMDB)

### Hardware
- Training performed on NVIDIA A100 GPUs with 80 GB memory

### Additional Notes
- The same hyperparameter settings are applied consistently across different missing modality scenarios  
- Ablation studies indicate a prompt pool size of 20 offers a good balance between accuracy and computational cost  
- FED-PRIME is optimized for lower GPU memory use by prepending prompts only to the first MSA layer

For further details, please refer to the full paper and supplementary material.
