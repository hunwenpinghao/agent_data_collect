# 评估批处理优化指南

## 概述

在模型评估时，合理使用批处理（batch processing）可以显著提高评估效率，充分利用GPU并行计算能力。

## 批处理的优势

### 🚀 性能提升
- **并行计算**: GPU可以同时处理多个样本
- **内存利用**: 更好地利用GPU显存
- **吞吐量**: 显著提高每秒处理的样本数

### 📊 效率对比
- `batch_size=1`: 逐个处理，GPU利用率低
- `batch_size=4`: 适中的并行度，平衡速度和内存
- `batch_size=8`: 更高的并行度，适合大显存GPU
- `batch_size=16+`: 高并行度，需要充足显存

## 批处理大小建议

### 根据GPU显存选择

| GPU显存 | 推荐batch_size | 模型大小 | 备注 |
|---------|---------------|----------|------|
| 4GB     | 1-2           | 小模型   | 保守设置 |
| 8GB     | 2-4           | 中等模型 | 平衡性能 |
| 12GB    | 4-8           | 大模型   | 良好性能 |
| 16GB+   | 8-16          | 大模型   | 高性能 |

### 根据模型大小选择

| 模型参数 | 推荐batch_size | 说明 |
|----------|---------------|------|
| < 1B     | 8-16          | 小模型，可以大批处理 |
| 1B-7B    | 4-8           | 中等模型 |
| 7B-13B   | 2-4           | 大模型，谨慎设置 |
| > 13B    | 1-2           | 超大模型，小批处理 |

## 使用方法

### 基本用法

```bash
# 默认批处理大小 (4)
./run_eval.sh -m model_path -d data_path

# 指定批处理大小
./run_eval.sh -m model_path -d data_path -b 8

# Python脚本直接调用
python eval_model.py -m model_path -d data_path --batch-size 8
```

### 性能测试

```bash
# 测试不同批处理大小的性能
for bs in 1 2 4 8 16; do
    echo "Testing batch_size=$bs"
    time ./run_eval.sh -m model_path -d data_path -b $bs -o eval_bs_$bs
done
```

## 优化建议

### 1. 渐进式测试
从小的batch_size开始，逐步增加：

```bash
# 从小开始测试
./run_eval.sh -m model_path -d data_path -b 2

# 如果没有OOM错误，尝试更大的
./run_eval.sh -m model_path -d data_path -b 4

# 继续增加直到达到最优
./run_eval.sh -m model_path -d data_path -b 8
```

### 2. 监控GPU使用

```bash
# 在另一个终端监控GPU使用
watch -n 1 nvidia-smi

# 或者使用gpustat
pip install gpustat
watch -n 1 gpustat
```

### 3. 内存优化设置

```bash
# 对于显存不足的情况
./run_eval.sh -m model_path -d data_path -b 2 --torch-dtype float16

# 或者强制使用CPU
./run_eval.sh -m model_path -d data_path -b 1 --device cpu
```

## 故障排除

### OOM (Out of Memory) 错误

**症状**: `CUDA out of memory` 或类似错误

**解决方案**:
1. 减小batch_size: `-b 2` 或 `-b 1`
2. 使用更低精度: `--torch-dtype float16`
3. 减少最大生成长度: `--max-tokens 256`
4. 使用CPU: `--device cpu`

```bash
# 保守设置
./run_eval.sh -m model_path -d data_path -b 1 --torch-dtype float16 --max-tokens 256
```

### 速度过慢

**症状**: 评估速度很慢

**解决方案**:
1. 增加batch_size: `-b 8` 或更高
2. 检查是否使用GPU: 确保 `--device auto`
3. 使用更快的数据类型: `--torch-dtype float16`

```bash
# 高性能设置
./run_eval.sh -m model_path -d data_path -b 8 --torch-dtype float16
```

### 批处理失败回退

如果批处理失败，系统会自动回退到单样本处理：

```
⚠️ 批量生成失败: CUDA out of memory
[INFO] 2024-01-15 14:30:25 - 自动回退到单样本生成
```

## 性能基准

### 典型性能提升

在配备8GB显存的GPU上测试100个样本：

| batch_size | 时间 | 提升 | GPU利用率 |
|------------|------|------|-----------|
| 1          | 120s | 基准 | ~30%      |
| 2          | 70s  | 1.7x | ~50%      |
| 4          | 45s  | 2.7x | ~70%      |
| 8          | 35s  | 3.4x | ~85%      |

### 最佳实践总结

1. **开始测试**: 使用默认的 `batch_size=4`
2. **监控资源**: 观察GPU利用率和显存使用
3. **逐步优化**: 如果资源充足，增加batch_size
4. **稳定运行**: 选择不出现OOM的最大batch_size
5. **记录设置**: 记录最佳配置用于后续评估

## 自动批处理优化脚本

可以创建一个自动寻找最佳batch_size的脚本：

```bash
#!/bin/bash
# find_optimal_batch_size.sh

MODEL_PATH="$1"
DATA_PATH="$2"

echo "寻找最佳batch_size..."

for bs in 1 2 4 8 16 32; do
    echo "测试 batch_size=$bs"
    
    # 测试是否会OOM
    timeout 60 ./run_eval.sh -m "$MODEL_PATH" -d "$DATA_PATH" -b $bs -o "test_bs_$bs" 2>&1 | grep -q "out of memory"
    
    if [ $? -eq 0 ]; then
        echo "batch_size=$bs 导致OOM，最佳值为 $((bs/2))"
        break
    else
        echo "batch_size=$bs 运行正常"
        BEST_BS=$bs
    fi
done

echo "推荐使用: batch_size=$BEST_BS"
```

---

通过合理使用批处理，你可以显著提高模型评估的效率！🚀 