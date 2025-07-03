# GPU内存优化配置指南

当遇到 `CUDA out of memory` 错误时，可以使用以下内存优化配置。

## 配置选择指南

### 1. 根据GPU内存选择配置

| GPU内存 | 推荐配置 | 命令 | 说明 |
|---------|----------|------|------|
| >12GB | `deepspeed` | `./run_train.sh -t deepspeed` | 标准ZeRO Stage 2 |
| 6-12GB | `stage2_offload` | `./run_train.sh -t stage2_offload` | ZeRO Stage 2 + CPU Offload |
| 4-6GB | `stage3` | `./run_train.sh -t stage3` | ZeRO Stage 3最大内存节省 |
| <4GB | `minimal` | `./run_train.sh -t minimal` | 紧急情况最小配置 |

### 2. 配置详细说明

#### 标准DeepSpeed配置 (`deepspeed`)
- **ZeRO Stage 2**: 参数和优化器状态分片
- **batch_size**: 4, **gradient_accumulation**: 4
- **序列长度**: 2048
- **适用**: GPU内存充足的情况

#### 平衡配置 (`stage2_offload`)
- **ZeRO Stage 2 + CPU Offload**: 优化器状态offload到CPU
- **batch_size**: 3, **gradient_accumulation**: 6
- **序列长度**: 1792
- **适用**: 中等GPU内存，需要平衡性能和内存

#### 最大内存优化 (`stage3`)
- **ZeRO Stage 3**: 参数、梯度、优化器状态全部分片
- **batch_size**: 2, **gradient_accumulation**: 8
- **序列长度**: 1536
- **适用**: GPU内存紧张但还能运行

#### 紧急配置 (`minimal`)
- **ZeRO Stage 3 + CPU Offload**: 最大程度内存节省
- **batch_size**: 1, **gradient_accumulation**: 16
- **序列长度**: 1024
- **适用**: GPU内存<4GB的极端情况

## 使用步骤

### 1. 首次遇到OOM时
```bash
# 从最激进的配置开始
./run_train.sh -t stage3
```

### 2. 如果stage3还是OOM
```bash
# 使用最小内存配置
./run_train.sh -t minimal
```

### 3. 如果还有余量，想要更好性能
```bash
# 尝试平衡配置
./run_train.sh -t stage2_offload
```

## 内存优化技巧

### 1. 运行前清理GPU内存
```bash
# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"

# 杀死占用GPU的进程
nvidia-smi
sudo kill -9 <PID>
```

### 2. 监控GPU内存使用
```bash
# 实时监控
watch -n 1 nvidia-smi

# 训练时监控
nvidia-smi dmon -s mu -d 1
```

### 3. 系统级优化
```bash
# 增加系统swap (临时)
sudo swapon --show
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 配置文件参数说明

### 关键内存优化参数
```json
{
  "per_device_train_batch_size": 1,    // 越小内存占用越少
  "gradient_accumulation_steps": 16,   // 补偿小batch size
  "max_seq_length": 1024,              // 序列长度影响内存占用
  "dataloader_num_workers": 0,         // 减少数据加载器内存
  "save_total_limit": 1,               // 减少保存的检查点数量
  "fp16": true,                        // 使用半精度浮点数
  "evaluation_strategy": "steps",      // 评估策略
  "eval_strategy": "steps",            // 兼容性参数（必需）
  "save_strategy": "steps",            // 保存策略
  "deepspeed_stage": 3,                // ZeRO优化级别
  "cpu_offload": true                  // CPU卸载优化器状态
}
```

### LoRA参数优化
```json
{
  "lora_r": 16,                        // 减少LoRA rank
  "lora_target_modules": "q_proj,k_proj,v_proj,o_proj",  // 减少目标模块
  "lora_alpha": 8                      // 适当减少alpha值
}
```

## 故障排除

### 1. 如果所有配置都OOM
- 检查是否有其他进程占用GPU
- 考虑使用更小的模型 (如0.5B而非1.8B)
- 减少序列长度到512或更小
- 关闭评估：`"evaluation_strategy": "no"`

### 2. 如果训练速度太慢
- 优先尝试`stage2_offload`而非`stage3`
- 增加`gradient_accumulation_steps`而非`batch_size`
- 确保有足够的CPU内存支持offload

### 3. 如果出现其他DeepSpeed错误
- 检查DeepSpeed版本：`pip list | grep deepspeed`
- 尝试重新安装：`pip install deepspeed --force-reinstall`
- 查看日志中的具体错误信息

## 性能对比

| 配置 | 内存使用 | 训练速度 | 效果质量 | 使用场景 |
|------|----------|----------|----------|----------|
| deepspeed | 高 | 快 | 最好 | 充足内存 |
| stage2_offload | 中高 | 中快 | 好 | 平衡选择 |
| stage3 | 中低 | 中慢 | 好 | 内存紧张 |
| minimal | 最低 | 最慢 | 可接受 | 紧急情况 |

记住：**训练能成功完成比速度更重要**！ 