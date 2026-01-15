# MeZO: Fine-Tuning Language Models with Just Forward Passes

**MeZO** is a memory-efficient zeroth-order optimizer for fine-tuning language models with just forward passes.

## Quick Start Guide

### Environment Setup

```bash
# Clone repository
git clone https://github.com/nyc030325/MeZO.git
cd MeZO

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Accept conda terms
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create and activate environment
conda create -n llama python=3.10 -y
conda activate llama

# Install dependencies
pip install -U transformers==4.28.1 datasets scikit-learn filelock tqdm numpy torch loralib
```

### Data Preparation

```bash
cd data
bash download_dataset.sh
cd ..

# Generate k-shot data for SST-2
python tools/generate_k_shot_data.py --k 16 --task SST-2 --seed 13 21 42 87 100 --mode k-shot-1k-test --data_dir data/original --output_dir data
python tools/generate_k_shot_data.py --k 512 --task SST-2 --seed 13 21 42 87 100 --mode k-shot-1k-test --data_dir data/original --output_dir data
```

### GPU Monitoring (Optional)

```bash
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 1 > gpu_mem.log
```

### Medium Models - MeZO
```bash
cd medium_models

# Run MeZO fine-tuning for K=16 and K=512
for K in 16 512; do for SEED in 13 21 42 87 100; do TASK=SST-2 K=$K SEED=$SEED BS=8 LR=1e-6 EPS=1e-3 STEP=100 EVAL_STEP=100 MODEL=roberta-large bash mezo.sh; done; done

# Gather results
python tools/gather_result.py --condition "{'tag': 'k16-roberta-large-mezo-ft', 'task_name': 'sst-2'}"
python tools/gather_result.py --condition "{'tag': 'k512-roberta-large-mezo-ft', 'task_name': 'sst-2'}"
```

### Medium Models - Adam

```bash
cd medium_models

# Run Adam fine-tuning for K=16 and K=512
for K in 16 512; do for SEED in 13 21 42 87 100; do TASK=SST-2 K=$K SEED=$SEED BS=8 LR=1e-5 STEP=100 EVAL_STEP=100 MODEL=roberta-large EXTRA_TAG=adam-step100-bs8-lr1e-5 bash finetune.sh; done; done

# Gather results
python tools/gather_result.py --condition "{'tag': 'k16-roberta-large-adam-step100-bs8-lr1e-5', 'task_name': 'sst-2'}"
python tools/gather_result.py --condition "{'tag': 'k512-roberta-large-adam-step100-bs8-lr1e-5', 'task_name': 'sst-2'}"
```

### Large Models (OPT-13B) - Experiments

```bash
cd large_models

# Install additional dependencies
pip install accelerate==0.17.1
mkdir result

# Run ICL with OPT-13B
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh --num_train 0
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh

# Run MeZO fine-tuning with OPT-13B
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash mezo.sh

# Run Adam fine-tuning with OPT-13B using FSDP
MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()") MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-5 NUM_GPU=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash finetune_fsdp.sh --overwrite_output_dir --save_strategy no --load_best_model_at_end False --save_total_limit 0 --save_model False --save_on_interrupt False
```
