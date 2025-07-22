# ICCV'25: Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data

This repository contains the official implementation of **FED-PRIME**, a novel federated prompt-tuning framework designed to handle **heterogeneous and incomplete multimodal data** in decentralized learning environments.

This work builds upon and extends Federated Learning concepts from the [easyFL](https://github.com/WwZzz/easyFL/tree/main) framework.  
**easyFL: A Lightning Framework for Federated Learning** , which provides the PyTorch implementation for the IJCAI'21 paper _Federated Learning with Fair Averaging_.


---

## Overview

FED-PRIME is a federated prompt-tuning framework designed to tackle the challenges of **heterogeneous and incomplete multimodal client data** common in real-world federated learning. Unlike existing methods, FED-PRIME handles both **inter-client heterogeneity** (different clients have varying missing modalities) and **intra-client heterogeneity** (missing patterns vary within client samples) in pretrained parameter-efficient fine-tuning settings.

The approach learns **two distinct prompt sets per client**:  
- **Inter-client prompts** capture shared missing modality patterns across clients, while  
- **Intra-client prompts** model client-specific, input-agnostic missingness.


Key strengths include robust handling of incomplete multimodal data, efficient fine-tuning without full model updates, and superior prompt alignment to boost federated learning performance under diverse missing data scenarios.

### Repository Structure
```bash
├── algorithm/                  # Server Aggregation Components
├── benchmark/                  # Clients' Model Architectures
├── notebook/                   # Data preprocess files
├── scripts/                    # Evaluation scripts
├── utils/                      # Utility files
├── DATA.md                     # Data readme files
├── env_setup.sh                # Set up environment to run
├── Experiments.md              # Details about experiment's hyperparameters 
├── generate_fedtask.py         # Create data files (client data partition) before running FL
├── main.py                     # Main running files
├── README.md                   # This file
├── requirements.txt            # Required Python packages
└── sweep_config.yaml           # Auto hyperparameter tuning
```

---

## Installation

```bash
   conda create -n fedprime python=3.8 -y
   conda activate fedprime
   bash env_setup.sh
   ```
---
## Usage

### Dataset Preparation

Prepare the multimodal datasets with simulated missing data patterns as described in the paper (e.g., UPMC Food-101, MM-IMDB). Note that both datasets are processed to include only the **8 most frequent classes** to simulate smaller client datasets typical in federated learning. Preprocessing scripts are included in the `data` folder.

We use two vision and language datasets: [MM-IMDb](https://github.com/johnarevalo/gmu-mmimdb), and [UPMC Food-101](https://visiir.isir.upmc.fr/explore). Please download the datasets by yourself. We use `pyarrow` to serialize the datasets. Please see `DATA.md` to organize the datasets, run the scripts in that file to prepare data.


### Training and Evaluation

For detailed experiment information, please refer to `Experiments.md`.

#### Step 1: Generate Federated Task (Client Data Partition)  
Run the following script to create federated data splits for 20 clients:

```bash
bash script/food101/main_exp_20clients/miss_both/20cl.sh
```

#### Step 2: Configure Weights & Biases (Wandb)  
You can customize Wandb settings by initializing it in your main script. For example:

```python
wandb.init(
    project="fed-prime", 
    entity="your-username", 
    name="your-experiment-name", 
    key="your-wandb-key"
)
```

#### Step 3: Run Experiments  
Execute the script below to start training across all specified models and algorithms:

```bash
bash script/food101/main_exp_20clients/miss_both/all_models.sh
```

This script will:  
- Generate federated learning data splits by calling `20cl.sh`.  
- Define which algorithms to run (default: "Fed-Intra").  
- For each algorithm, run `main.py` with your training settings (task, model, learning rate, epochs, batch size, GPU, etc.).  
- Log results to Weights & Biases and display progress updates.


Additional scripts and resources are available in the `./script/` directory.


### Automatic Hyperparameter Tuning with W&B Sweeps

To perform automatic hyperparameter tuning, use Weights & Biases (W&B) sweep functionality:

#### 1. Create the Federated Task for Your Experiment  
Generate the federated task for the experiment you want to tune.  
For example, to create a fedtask with 20 clients under the "Miss Both" scenario and missing ratios of 0.7 for both train and test sets, run:

```bash
python generate_fedtask.py \
    --benchmark food101 \
    --dist 0 \
    --skew 0 \
    --num_clients 20 \
    --seed 0 \
    --missing \
    --missing_ratio_train 0.7 \
    --missing_ratio_test 0.7 \
    --missing_type_train both \
    --missing_type_test both \
    --both_ratio 0.5 \
    --max_text_len 40
```

#### 2. Create and Start the Sweep  
Modify the sweep configuration file to specify which hyperparameters to tune automatically.  
Example sweep configuration snippet (`sweep_config.yaml`):

```yaml
program: main.py

method: bayes

metric:
  name: test_acc.max
  goal: maximize

parameters:
  pool_size:
    values: [10]
  top_k:
    values: [2,4]
  num_outer_loops:
    values: [5]
```

Set the appropriate command-line arguments for your experiment in the sweep config. For example:

```yaml
args:
  - "--task"
  - "food101_cnum20_dist0_skew0_seed0_missing_ratio_0.7_0.7_missing_type_both_both_both_ratio_0.5"
  - "--model"
  - "Fed-Prime"
  - "--algorithm"
  - "multimodal.food101.Fed-Prime"
```

Start the sweep with this command:

```bash
wandb sweep sweep_config.yaml
```

You will see output similar to:

```
wandb: Creating sweep from: sweep_config.yaml
wandb: Creating sweep with ID: v6xu1o2b
wandb: View sweep at: <link>
wandb: Run sweep agent with: wandb agent your_wandb_username/v6xu1o2b
```

#### 3. Run the Sweep Agent  
Use the sweep ID from the previous step to start the sweep agent, which runs experiments with different hyperparameter settings:

```bash
wandb agent entity_name/project_name/v6xu1o2b
```


This will automatically perform hyperparameter tuning over the defined parameter space using W&B Sweeps.

---



### Results
FED-PRIME achieves state-of-the-art performance in multimodal federated learning with heterogeneous missing data, outperforming adapted baselines by up to 107% relative improvement on classification accuracy/F1 scores across extensive experiments.

### Key findings include:

Robustness to missing data with varying missing rates.

Superior embedding alignment across incomplete multimodal inputs.

Efficient training with reduced GPU memory usage compared to baseline methods.

For detailed experimental results and analysis, please refer to the paper and supplementary material.
