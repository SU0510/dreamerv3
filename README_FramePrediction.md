# Frame Prediction RSSM Project

## 项目结构

```
frame_prediction_rssm/
├── __init__.py              # 包初始化
├── rssm_core.py            # RSSM核心模块
├── encoder.py              # 帧编码器
├── decoder.py              # 帧解码器+点云生成
├── pipeline.py             # 完整管道
├── training.py             # 训练/推理工具
├── config.py               # 配置预设
└── README.md               # 详细文档

examples/
├── train_frame_prediction.py  # 训练演示脚本
├── inference.py               # 推理脚本
└── notebook_example.py        # Jupyter示例
```

## 快速开始

### 安装

```bash
cd frame_prediction_rssm
pip install -r requirements.txt
```

### 基本用法

```python
import jax
import jax.numpy as jnp
from frame_prediction_rssm.pipeline import FramePredictionPipeline
from frame_prediction_rssm.training import create_train_state, predict

# 初始化模型
rng = jax.random.PRNGKey(42)
train_state, apply_fn = create_train_state(rng, learning_rate=1e-4)

# 准备输入：前4帧
frames_1to4 = jnp.zeros((batch_size, 4, 64, 64, 3), dtype=jnp.uint8)

# 推理：预测第5帧并生成点云
predicted_frame, point_cloud = predict(train_state, frames_1to4, apply_fn)

# 点云包含：
# - positions: [B, H*W, 3] XYZ坐标
# - features: [B, H*W, feature_dim] 稠密特征
# - confidences: [B, H*W] 置信度
```

### 运行示例

```bash
# 训练示例
cd examples
python train_frame_prediction.py

# 推理示例
python inference.py --frame_dir ./frames --model_size small --output_dir ./outputs
```

## 核心概念

### 1. RSSM隐空间预测

RSSM (Recurrent State Space Model) 维护两种状态：

- **确定性状态** `deter`: 记录所有信息，维度大 (256-512)
- **随机状态** `stoch`: 离散隐变量集合，维度 (stoch_size, classes)

预测流程：
```
输入帧 1-4
    ↓
编码为 tokens
    ↓
初始化 RSSM 状态 (deter=0, stoch=0)
    ↓
逐帧处理 (4步 RSSM 更新)
    ↓
最终状态编码了整个序列信息
    ↓
预测第5帧
```

### 2. 稠密特征生成

从 RSSM 隐状态生成稠密特征的过程：

```
RSSM State (deter [256], stoch [32×32])
    ↓
连接并投影到空间分辨率 (16×16×256)
    ↓
预测深度图、置信度、特征
    ↓
生成 XY 网格坐标
    ↓
4D 点云 (X, Y, Z, Features)
```

### 3. 4D点云格式

每个点包含：
- **位置 (X, Y, Z)**: 归一化坐标 + 深度预测
- **特征 (256维)**: 稠密特征向量
- **置信度 (0-1)**: 该点的有效性分数

## 架构组件详解

### Encoder (encoder.py)

- `FrameEncoder`: CNN处理单帧，输出 [B, T, latent_dim]
- `TemporalEncoder`: 多头自注意力捕捉时间关系
- `MultiFrameEncoder`: 组合视觉和时间编码

输出：[batch, time, latent_dim] frame tokens

### RSSM (rssm_core.py)

- `RSSMCore`: 更新确定性状态 $h_{t+1} = f(h_t, z_t, a_t)$
- `StochasticEncoder`: 从确定性状态编码随机状态
- `RSSM`: 单步预测 
- `RSSMSequence`: 序列处理 (4步输入 + 1步预测)

### Decoder (decoder.py)

- `FeatureDecoder`: 将 RSSM 状态解码为 RGB 帧
- `DenseFeatureExtractor`: 提取空间特征图 [B, H, W, F]
- `PointCloudGenerator`: 生成完整的 4D 点云
- `AdaptivePointCloudGenerator`: 自适应点云 (可过滤低置信度点)

### Pipeline (pipeline.py)

完整的端到端处理：
```
frames_1to4 [B,4,H,W,3]
    ↓ (MultiFrameEncoder)
frame_tokens [B,4,latent_dim]
    ↓ (RSSMSequence x 4 steps)
states [4个RSSMState]
    ↓ (RSSM x 1 step predicted)
predicted_state
    ↓ (FeatureDecoder + PointCloudGenerator)
predicted_frame [B,H,W,3]
point_cloud {positions, features, confidences}
```

## 训练

### 损失函数

```python
Total Loss = L_frame + λ_pc × L_pc

L_frame = MSE(predicted_frame, target_frame)

L_pc = λ_conf × Entropy(confidences) + λ_depth × Smooth(depth)
```

### 训练循环

```python
from frame_prediction_rssm.training import train_step, eval_step

for epoch in range(num_epochs):
    for batch in train_loader:
        train_state, metrics = train_step(
            train_state,
            batch['frames_1to4'],
            batch['target_frame_5'],
            apply_fn
        )
    
    # 评估
    eval_metrics = eval_step(...)
```

## 配置预设

### 预定义配置

```python
from frame_prediction_rssm.config import get_config

# 选择模型大小
config = get_config('small')  # 'tiny', 'small', 'medium', 'large'
```

参数数量和计算量：
- **tiny**: ~250K 参数 (快速原型)
- **small**: ~2.5M 参数 (推荐)
- **medium**: ~12M 参数 (高质量)
- **large**: ~60M 参数 (最优结果)

## 使用场景

### 1. 视频帧插值
预测中间帧以进行平滑视频：
```python
# 给定帧 1-4，预测帧 5
# 循环使用预测结果作为输入的最后一帧
```

### 2. 点云预测
从视频序列生成 3D 点云：
```python
point_cloud = predict(...)[1]
# 导出为 PLY 或其他 3D 格式
```

### 3. 特征提取
提取稠密特征用于下游任务：
```python
features = point_cloud['features']  # [B, 256, feature_dim]
# 用于 3D 检测、分割等
```

## 进阶用法

### 自定义损失函数

```python
def custom_loss(predicted, target):
    # 自己的损失定义
    return loss

# 在 training.py 中修改 train_step
```

### 微调预训练模型

```python
# 加载预训练权重
with open('pretrained.pkl', 'rb') as f:
    pretrained_params = pickle.load(f)

train_state = train_state.replace(params=pretrained_params)

# 继续训练 (较低学习率)
train_state, _ = train_step(train_state, ...)
```

### 多GPU训练

```python
# 使用 flax.jax 的 pmap
from jax import pmap

@pmap
def train_step_pmap(state, batch):
    return train_step(state, batch['frames'], batch['target'], apply_fn)

# 将数据分配到 GPU
for batch in batches:
    batch = jax.tree_util.tree_map(lambda x: x.reshape(n_devices, -1, ...), batch)
    states = train_step_pmap(states, batch)
```

## 可视化

### 可视化预测帧

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i in range(4):
    axes[i].imshow(frames_1to4[0, i])
    axes[i].set_title(f'Frame {i+1}')

axes[4].imshow(predicted_frame[0])
axes[4].set_title('Predicted Frame 5')

plt.tight_layout()
plt.show()
```

### 可视化点云

```python
# 导出为 PLY 格式 (在 inference.py 中已实现)
# 使用 CloudCompare 或其他工具打开 .ply 文件

# 或用 plotly 3D 绘图
import plotly.graph_objects as go

positions = point_cloud['positions'][0]
confidences = point_cloud['confidences'][0]

scatter = go.Scatter3d(
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        color=confidences,
        colorscale='Viridis'
    )
)

fig = go.Figure(data=[scatter])
fig.show()
```

## 常见问题

**Q: 如何处理不同分辨率的帧？**
A: 在 encoder 中添加自适应池化或在管道前调整大小。

**Q: 可以预测多步前的帧吗？**
A: 可以！循环使用预测作为下一步的输入：
```python
frame_5_pred = predict(frames_1to4)[0]
frames_2to5 = [frames_1to4[:, 1:], frame_5_pred]
frame_6_pred = predict(frames_2to5)[0]
```

**Q: 如何改进点云质量？**
A: 
- 增大 `pc_spatial_height/width` 来生成更密集的点云
- 增加 `pc_feature_dim` 提升特征表现力
- 调整 `confidence_threshold` 过滤低质量点

**Q: 支持条件预测吗？**
A: 支持！在 RSSM 中添加条件向量：
```python
# 修改 RSSM._core 接收条件
conditional_rssm(deter, stoch, action, condition)
```

## 参考文献

- **[Dreamer V3]** Hafner et al., "Mastering Diverse Domains through World Models"
  https://arxiv.org/abs/2301.04104
  
- **[Video Prediction with RSSM]** 原始 RSSM 架构
  
- **[3D Point Cloud Generation]** 从隐空间生成 3D 表示

## 许可证

遵循原 Dreamer V3 项目的许可制

## 作者

基于 Dreamer V3 实现的简化版本，用于帧预测和点云生成

## 致谢

感谢 Dreamer V3 的作者们提供的灵感和架构参考！
