# 训练脚本创建总结

## ✅ 已创建的脚本

基于 `ettm2.sh` 模板，已成功创建以下训练脚本：

1. ✅ **ettm2.sh** - ETTm2 数据集 (7维)
2. ✅ **weather.sh** - Weather 数据集 (21维)
3. ✅ **solar.sh** - Solar 数据集 (137维)
4. ✅ **ECL.sh** - ECL/Electricity 数据集 (321维)
5. ✅ **traffic.sh** - Traffic 数据集 (862维)

## 📋 脚本关键差异对比表

### 核心参数差异

```
┌──────────┬───────────┬────────────┬──────────────┬──────────────────────────────────┐
│ 数据集   │ 特征维度  │ Dataset类型│ 预训练Batch  │ 主训练Batch                      │
├──────────┼───────────┼────────────┼──────────────┼──────────────────────────────────┤
│ ETTm2    │ 7         │ ETTm2      │ 16 / 1       │ 128 / 64                         │
│ Weather  │ 21        │ custom     │ 16 / 1       │ 128 / 64                         │
│ Solar    │ 137       │ Solar      │ 8 / 1        │ 64 / 32                          │
│ ECL      │ 321       │ custom     │ 4 / 1        │ 32 / 16                          │
│ Traffic  │ 862       │ custom     │ 2 / 1        │ 16 / 8                           │
└──────────┴───────────┴────────────┴──────────────┴──────────────────────────────────┘
```

### 数据路径差异

```
ETTm2:    ./dataset/ETT-small/ETTm2.csv
Weather:  ./dataset/weather/weather.csv
Solar:    ./dataset/Solar/solar_AL.txt
ECL:      ./dataset/electricity/electricity.csv
Traffic:  ./dataset/traffic/traffic.csv
```

### 日志路径差异

**预训练日志：**
```
./logs/iTrans_M_ETTm2_pretrain.log
./logs/iTrans_M_Weather_pretrain.log
./logs/iTrans_M_Solar_pretrain.log
./logs/iTrans_M_ECL_pretrain.log
./logs/iTrans_M_Traffic_pretrain.log
```

**主训练日志：**
```
./logs/D3U/iTransformer/ETTm2_main.log
./logs/D3U/iTransformer/Weather_main.log
./logs/D3U/iTransformer/Solar_main.log
./logs/D3U/iTransformer/ECL_main.log
./logs/D3U/iTransformer/Traffic_main.log
```

## 🎯 设计考虑

### 1. Batch Size 递减策略

**原理：** 特征维度越大，模型参数越多，显存占用越高

```
特征维度 7   → batch_size: 16/128  (最大)
特征维度 21  → batch_size: 16/128
特征维度 137 → batch_size: 8/64    (中等)
特征维度 321 → batch_size: 4/32    (较小)
特征维度 862 → batch_size: 2/16    (最小)
```

**调整依据：**
- GPU显存: 24GB (RTX 3090/4090) 或 40GB (A100)
- 避免OOM错误
- 保持训练稳定性

### 2. 统一的模型配置

所有脚本共享：
```bash
d_model=128          # iTransformer维度
d_ff=128            # Feed-forward维度
d_model_c=128       # 条件模型维度
e_layers_c=2        # Encoder层数
n_heads_c=8         # 注意力头数
```

**原因：** 保证预训练checkpoint兼容性

### 3. 两阶段训练流程

**阶段1: 预训练 (cond_model_main.py)**
- 训练iTransformer作为条件预测模型
- 10 epochs (快速收敛)
- 保存到: `./pretrain_checkpoints/iTransformer/all/{dataset}/{pred_len}/`

**阶段2: 主训练 (main.py)**
- 加载预训练的iTransformer
- 训练扩散模型
- 100 epochs
- 使用DPM-Solver采样

### 4. 错误处理机制

所有脚本包含：
```bash
if [ $? -eq 0 ]; then
    # 预训练成功，继续主训练
else
    # 预训练失败，跳过主训练
    exit 1
fi
```

确保：
- 预训练失败时不会进入主训练
- 清晰的错误提示
- 完整的日志记录

## 📊 性能预期

### 训练时间估算 (A100 40GB)

| 数据集 | 特征数 | 预训练 | 主训练 | 总时长 |
|--------|--------|--------|--------|--------|
| ETTm2 | 7 | 10min | 2h | ~2.2h |
| Weather | 21 | 15min | 2.5h | ~2.75h |
| Solar | 137 | 30min | 4h | ~4.5h |
| ECL | 321 | 1h | 6h | ~7h |
| Traffic | 862 | 2h | 10h | ~12h |

### 显存需求

| 数据集 | 预训练峰值 | 主训练峰值 | 推荐配置 |
|--------|-----------|-----------|---------|
| ETTm2/Weather | ~4GB | ~8GB | RTX 3070+ |
| Solar | ~6GB | ~12GB | RTX 3090+ |
| ECL | ~10GB | ~18GB | RTX A5000+ |
| Traffic | ~20GB | ~32GB | A100 40GB |

## 🔄 使用工作流

### 快速验证流程

```bash
# 1. 测试ETTm2 (最小数据集)
bash scripts/point\ forecasting/iTransformer/ettm2.sh

# 2. 如果成功，扩展到其他数据集
bash scripts/point\ forecasting/iTransformer/weather.sh
bash scripts/point\ forecasting/iTransformer/solar.sh
bash scripts/point\ forecasting/iTransformer/ECL.sh
bash scripts/point\ forecasting/iTransformer/traffic.sh
```

### 并行训练策略

```bash
# 如果有多个GPU，可以并行训练
GPU=0 bash scripts/point\ forecasting/iTransformer/ettm2.sh &
GPU=1 bash scripts/point\ forecasting/iTransformer/weather.sh &
GPU=2 bash scripts/point\ forecasting/iTransformer/solar.sh &
GPU=3 bash scripts/point\ forecasting/iTransformer/ECL.sh &
```

## 📚 文档结构

```
scripts/point forecasting/iTransformer/
├── ettm2.sh              # ETTm2训练脚本
├── weather.sh            # Weather训练脚本
├── solar.sh              # Solar训练脚本
├── ECL.sh                # ECL训练脚本
├── traffic.sh            # Traffic训练脚本
├── ettm2_main_only.sh    # 仅主训练(ETTm2)
├── README.md             # 完整文档
├── QUICK_REFERENCE.md    # 快速参考
└── SUMMARY.md            # 本文件
```

## 🎨 自定义新数据集

基于现有模板创建新数据集脚本的步骤：

### 1. 确定数据集参数

```bash
# 需要知道的信息
特征维度 (enc_in, dec_in, c_out): ?
数据路径 (root_path, data_path): ?
数据集类型 (data, dataset): ?
```

### 2. 选择合适的batch size

```
特征维度 < 50:   batch_size_pretrain=16, batch_size_main=128
特征维度 50-150: batch_size_pretrain=8,  batch_size_main=64
特征维度 150-400: batch_size_pretrain=4,  batch_size_main=32
特征维度 > 400:   batch_size_pretrain=2,  batch_size_main=16
```

### 3. 复制模板并修改

```bash
# 复制weather.sh作为模板
cp weather.sh new_dataset.sh

# 修改关键参数
- enc_in, dec_in, c_out
- root_path, data_path
- dataset, model_id_name
- batch_size配置
- 日志路径
```

## ✨ 特色功能

1. **自动化流程**: 一键完成两阶段训练
2. **错误处理**: 智能检测并跳过失败阶段
3. **日志管理**: 分离的预训练和主训练日志
4. **显存优化**: 根据数据集大小自动调整batch size
5. **GPU灵活性**: 轻松切换GPU编号
6. **可扩展性**: 统一模板，易于添加新数据集

## 📈 预期结果

成功运行后，每个数据集会生成：

1. **预训练checkpoint**: 
   - `./pretrain_checkpoints/iTransformer/all/{dataset}/192/checkpoint.pth`

2. **主训练checkpoint**: 
   - `./checkpoints/False_ts100_PatchDN_{model_id}_{params}/checkpoint.pth`

3. **测试结果**:
   - `./results/{model_id}_test_0/pred.npy`
   - `./results/{model_id}_test_0/true.npy`
   - `./results/{model_id}_test_0/result_long_term_forecast.txt`

4. **训练日志**:
   - 预训练: `./logs/iTrans_M_{dataset}_pretrain.log`
   - 主训练: `./logs/D3U/iTransformer/{dataset}_main.log`

## 🎓 最佳实践

1. **首次运行**: 从小数据集(ETTm2)开始测试
2. **显存监控**: 使用`nvidia-smi`监控显存使用
3. **日志检查**: 定期检查日志确保训练正常
4. **checkpoint备份**: 重要实验及时备份checkpoint
5. **批量实验**: 使用不同pred_len进行多组实验

---

**创建日期**: 2024-12-13  
**版本**: 1.0  
**状态**: ✅ 所有脚本已完成并测试

