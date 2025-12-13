# GPU追踪报告：gpu_id=2 的使用情况

## 📋 概述
追踪 `gpu_id=2` 在预训练和主训练两个阶段中的完整传递和使用逻辑。

---

## 🔄 第一阶段：预训练（cond_model_main.py → iTransformer）

### 1️⃣ **参数传递**
```bash
# scripts/point forecasting/iTransformer/ettm2.sh 第50行
gpu_id=2

# 第93行传递给 cond_model_main.py
--gpu $gpu_id  # 实际值：--gpu 2
```

### 2️⃣ **参数解析**
```python
# cond_model_main.py 第111行
parser.add_argument('--gpu', type=int, default=0, help='gpu')

# 第115行
args = parser.parse_args()  # args.gpu = 2
```

### 3️⃣ **设备初始化**
```python
# models/exp/exp_basic_point.py 第23-32行
def _acquire_device(self):
    if self.args.use_gpu:
        # 关键步骤1: 设置环境变量，限制可见GPU为物理GPU 2
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)  # "2"
        
        # 关键步骤2: 创建逻辑设备 cuda:0 (映射到物理GPU 2)
        device = torch.device('cuda:0')
        
        print('Use GPU: cuda:{}'.format(self.args.gpu))  # 打印: Use GPU: cuda:2
    return device

# 第16行: 模型移动到设备
self.model = self._build_model().to(self.device)  # 模型在物理GPU 2上
```

**🔑 关键机制：CUDA_VISIBLE_DEVICES**
- `os.environ["CUDA_VISIBLE_DEVICES"] = "2"` 使得程序只能看到物理GPU 2
- PyTorch中的 `cuda:0` 被映射到物理GPU 2
- 这是一种标准的GPU隔离技术

### 4️⃣ **训练阶段数据传输**
```python
# models/exp/exp_long_term_forecasting_point.py 第129-136行
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    batch_x = batch_x.float().to(self.device)        # ✅ 移动到GPU 2
    batch_y = batch_y.float().to(self.device)        # ✅ 移动到GPU 2
    batch_x_mark = batch_x_mark.float().to(self.device)  # ✅ 移动到GPU 2
    batch_y_mark = batch_y_mark.float().to(self.device)  # ✅ 移动到GPU 2
    dec_inp = torch.cat([...]).float().to(self.device)   # ✅ 移动到GPU 2
```

### 5️⃣ **验证阶段数据传输**
```python
# models/exp/exp_long_term_forecasting_point.py 第56-80行
def vali(self, vali_data, vali_loader, criterion):
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
        batch_x = batch_x.float().to(self.device)        # ✅ 移动到GPU 2
        batch_x_mark = batch_x_mark.float().to(self.device)  # ✅ 移动到GPU 2
        batch_y_mark = batch_y_mark.float().to(self.device)  # ✅ 移动到GPU 2
        dec_inp = torch.cat([...]).float().to(self.device)   # ✅ 移动到GPU 2
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)  # ✅ 移动到GPU 2
```

### 6️⃣ **测试阶段数据传输**
```python
# models/exp/exp_long_term_forecasting_point.py 第220-228行
def test(self, setting, test=0, save_result=False, plot=False):
    # 第206行: 模型确保在正确设备上
    self.model.to(self.device)  # ✅ 模型在GPU 2上
    
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        batch_x = batch_x.float().to(self.device)        # ✅ 移动到GPU 2
        batch_y = batch_y.float().to(self.device)        # ✅ 移动到GPU 2
        batch_x_mark = batch_x_mark.float().to(self.device)  # ✅ 移动到GPU 2
        batch_y_mark = batch_y_mark.float().to(self.device)  # ✅ 移动到GPU 2
```

**✅ 预训练阶段结论：所有操作（训练、验证、测试）都正确使用GPU 2**

---

## 🔄 第二阶段：主训练（main.py → 扩散模型）

### 1️⃣ **参数传递**
```bash
# scripts/point forecasting/iTransformer/ettm2.sh 第148行
--gpu $gpu_id  # 实际值：--gpu 2
```

### 2️⃣ **参数解析**
```python
# utils/params_init.py (由main.py调用)
parser.add_argument('--gpu', type=int, default=1, help='gpu')
args = params_init.get_args()  # args.gpu = 2
```

### 3️⃣ **GPU设备设置**
```python
# main.py 第25-33行
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu:
    if args.use_multi_gpu:
        # 多GPU模式
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        # 单GPU模式：直接设置当前设备为GPU 2
        torch.cuda.set_device(args.gpu)  # ✅ 设置为GPU 2
```

### 4️⃣ **设备初始化**
```python
# models/exp/exp_basic.py 第24-37行
def _acquire_device(self):
    if torch.cuda.is_available():
        if self.args.use_gpu:
            # 关键步骤1: 设置环境变量
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)  # "2"
            
            # 关键步骤2: 创建逻辑设备
            device = torch.device('cuda:0')  # 映射到物理GPU 2
            
            print('Use GPU: cuda:{}'.format(self.args.gpu))  # 打印: Use GPU: cuda:2
    return device

# 第16-17行: 模型移动到设备
self.model = model.to(self.device)  # 扩散模型在GPU 2上
self.cond_pred_model = cond_pred_model.to(self.device)  # iTransformer在GPU 2上
```

### 5️⃣ **训练阶段数据传输**
```python
# models/exp/exp_main.py 第246-262行
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
    batch_x = batch_x.float().to(self.device)        # ✅ 移动到GPU 2
    batch_y = batch_y.float().to(self.device)        # ✅ 移动到GPU 2
    batch_x_mark = batch_x_mark.float().to(self.device)  # ✅ 移动到GPU 2
    batch_y_mark = batch_y_mark.float().to(self.device)  # ✅ 移动到GPU 2
    dec_inp = torch.cat([...]).float().to(self.device)   # ✅ 移动到GPU 2
    
    # 时间步也在GPU 2上
    t = torch.randint(low=0, high=self.model.num_timesteps, size=(n // 2 + 1,)).to(self.device)
    t = torch.cat([t, self.model.num_timesteps - 1 - t], dim=0)[:n].to(self.device)
```

### 6️⃣ **验证和测试阶段**
```python
# models/exp/exp_main.py 测试方法中
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
    batch_x = batch_x.float().to(self.device)        # ✅ 移动到GPU 2
    batch_y = batch_y.float().to(self.device)        # ✅ 移动到GPU 2
    # ... 所有数据都移动到self.device (GPU 2)
```

**✅ 主训练阶段结论：所有操作（训练、验证、测试）都正确使用GPU 2**

---

## 🎯 总结

### ✅ **确认：GPU 2 被正确使用**

| 阶段 | 训练 | 验证 | 测试 | 设备设置方式 |
|------|------|------|------|-------------|
| **预训练（iTransformer）** | ✅ GPU 2 | ✅ GPU 2 | ✅ GPU 2 | `CUDA_VISIBLE_DEVICES="2"` + `cuda:0` |
| **主训练（扩散模型）** | ✅ GPU 2 | ✅ GPU 2 | ✅ GPU 2 | `torch.cuda.set_device(2)` + `CUDA_VISIBLE_DEVICES="2"` + `cuda:0` |

### 🔑 **关键机制**

1. **环境变量隔离**：
   ```python
   os.environ["CUDA_VISIBLE_DEVICES"] = "2"
   ```
   - 限制程序只能看到物理GPU 2
   - 提供GPU资源隔离

2. **设备映射**：
   ```python
   device = torch.device('cuda:0')  # 逻辑GPU 0 → 物理GPU 2
   ```
   - 在程序内部使用 `cuda:0`
   - 实际运行在物理GPU 2上

3. **显式设备设置（仅主训练）**：
   ```python
   torch.cuda.set_device(2)  # 直接设置当前CUDA设备
   ```

### 📊 **数据流追踪**

```
脚本参数 (gpu_id=2)
    ↓
命令行参数 (--gpu 2)
    ↓
args.gpu = 2
    ↓
CUDA_VISIBLE_DEVICES="2" + torch.cuda.set_device(2)
    ↓
self.device = torch.device('cuda:0') [映射到物理GPU 2]
    ↓
模型: .to(self.device)
    ↓
数据: batch_x.to(self.device), batch_y.to(self.device), ...
    ↓
✅ 所有计算在物理GPU 2上执行
```

### ⚡ **验证方法**

可以在运行时使用以下命令验证GPU使用情况：

```bash
# 实时监控GPU 2的使用情况
watch -n 1 nvidia-smi

# 或者只查看GPU 2
nvidia-smi -i 2

# 在训练开始后，应该看到GPU 2的显存占用和利用率上升
```

### 🎉 **最终结论**

**是的！在预训练和主训练的所有阶段（训练、验证、测试）中，都正确使用了GPU 2进行计算。**

整个流程通过：
1. 环境变量 `CUDA_VISIBLE_DEVICES`
2. 设备对象 `self.device = torch.device('cuda:0')`
3. 数据和模型的 `.to(self.device)` 调用

确保了所有张量操作都在指定的GPU 2上执行。

