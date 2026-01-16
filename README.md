# MeZO: 仅需前向传播即可微调语言模型

**MeZO** 是一种内存高效的零阶优化器，它允许仅通过前向传播（Forward Passes）来微调语言模型。

## 快速开始指南

### 1. 环境配置

```bash
# 克隆代码仓库
git clone https://github.com/nyc030325/MeZO.git  
cd MeZO

# 安装 Miniconda (如果尚未安装)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
bash Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc

# 接受 Conda 许可条款（如有必要）
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main  
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r  

# 创建并激活名为 Mezo 的 Python 3.10 虚拟环境
conda create -n Mezo python=3.10 -y
conda activate Mezo

# 安装项目所需的 Python 依赖包
pip install -U transformers==4.28.1 datasets scikit-learn filelock tqdm numpy torch loralib
```

### 2. 数据准备
```bash
cd medium_models
cd data
# 下载基础数据集
bash download_dataset.sh
cd ..

# 为 SST-2 任务生成 K-shot 数据集 (K=16 和 K=512)
# 分别在 5 个随机种子 (13, 21, 42, 87, 100) 下生成数据
python tools/generate_k_shot_data.py --k 16 --task SST-2 --seed 13 21 42 87 100 --mode k-shot-1k-test --data_dir data/original --output_dir data
python tools/generate_k_shot_data.py --k 512 --task SST-2 --seed 13 21 42 87 100 --mode k-shot-1k-test --data_dir data/original --output_dir data
```


### 3. GPU 监控 (可选)
```bash
# 开启后台进程记录 GPU 显存使用情况
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 1 > gpu_mem.log
```

### 4. 中等规模模型实验 (RoBERTa-large)
```bash
# --- MeZO 微调实验 ---
# 运行 MeZO：遍历 K=16 和 K=512 两种设置，以及 5 个随机种子
for K in 16 512; do for SEED in 13 21 42 87 100; do TASK=SST-2 K=$K SEED=$SEED BS=64 LR=1e-6 EPS=1e-3 STEP=100 EVAL_STEP=100 MODEL=roberta-large bash mezo.sh; done; done

# 汇总 MeZO 实验结果
python tools/gather_result.py --condition "{'tag': 'k16-roberta-large-mezo-ft', 'task_name': 'sst-2'}"
python tools/gather_result.py --condition "{'tag': 'k512-roberta-large-mezo-ft', 'task_name': 'sst-2'}"

# --- Adam 对比实验 ---
# 运行 Adam（作为基准）：同样遍历 K=16 和 K=512 及 5 个随机种子
for K in 16 512; do for SEED in 13 21 42 87 100; do TASK=SST-2 K=$K SEED=$SEED BS=8 LR=1e-5 STEP=100 EVAL_STEP=100 MODEL=roberta-large EXTRA_TAG=adam-step100-bs8-lr1e-5 bash finetune.sh; done; done

# 汇总 Adam 实验结果
python tools/gather_result.py --condition "{'tag': 'k16-roberta-large-adam-step100-bs8-lr1e-5', 'task_name': 'sst-2'}"
python tools/gather_result.py --condition "{'tag': 'k512-roberta-large-adam-step100-bs8-lr1e-5', 'task_name': 'sst-2'}"
```

### 5. 大规模模型实验 (OPT-13B)
```bash
cd ..
cd large_models

# 安装大模型训练所需的额外依赖 (accelerate)
pip install accelerate==0.17.1
mkdir result

# 1. 零样本学习 (Zero-shot) 基准测试
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh --num_train 0

# 2. 上下文学习 (In-context learning, 32-shot) 基准测试
MODEL=facebook/opt-13b TASK=SST2 bash icl.sh 

# 3. 使用全分片数据并行 (FSDP) 进行全参数微调 (Adam 优化器，多 GPU 环境)
# 注意：此命令需要多卡环境支持
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-5 NUM_GPU=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash finetune_fsdp.sh --overwrite_output_dir --save_strategy no --load_best_model_at_end False --save_total_limit 0 --save_model False --save_on_interrupt False

# 4. MeZO 微调实验 (包含全参数、Prefix-tuning 和 LoRA 三种模式)
# 全参数微调 (Full-parameter)
MODEL=facebook/opt-13b TASK=SST2 MODE=ft LR=1e-7 EPS=1e-3 bash mezo.sh
# 前缀微调 (Prefix-tuning)
MODEL=facebook/opt-13b TASK=SST2 MODE=prefix LR=1e-3 EPS=1e-1 bash mezo.sh
# LoRA 微调
MODEL=facebook/opt-13b TASK=SST2 MODE=lora LR=5e-5 EPS=1e-2 bash mezo.sh
```
