# Frame Prediction with RSSM

基于 Dreamer V3 思想的简单帧预测实现，从前4帧预测第5帧并生成稠密特征的4D点云。

## 项目概述

该项目实现了一个完整的帧预测管道：

```
Frame 1-4 → Encoder → RSSM → Decoder + Point Cloud Generator → Frame 5 + 4D Point Cloud
```

### 核心组件

1. **Encoder** (`encoder.py`)
   - `FrameEncoder`: CNN视觉编码器，处理单帧
   - `TemporalEncoder`: 时间注意力模块，捕捉帧间时间关系
   - `MultiFrameEncoder`: 结合视觉和时间编码

2. **RSSM** (`rssm_core.py`) - 递归状态空间模型
   - `RSSMState`: 状态表示 (确定性状态 + 随机状态)
   - `RSSMCore`: 核心动态模型
   - `StochasticEncoder`: 隐空间编码器
   - `RSSM`: 完整RSSM模块
   - `RSSMSequence`: 序列处理

3. **Decoder** (`decoder.py`)
   - `FeatureDecoder`: 从潜在状态重建帧
   - `DenseFeatureExtractor`: 提取稠密特征图
   - `PointCloudGenerator`: 生成4D点云
   - `AdaptivePointCloudGenerator`: 自适应点云生成

4. **Pipeline** (`pipeline.py`)
   - `FramePredictionPipeline`: 完整端到端管道
   - `InferenceModel`: 推理模型包装器

5. **Training** (`training.py`)
   - 训练状态初始化
   - 损失函数计算
   - 训练和评估步骤

## 安装

```bash
# 安装依赖
pip install jax jaxlib flax optax numpy

# 可选：安装PIL用于图像处理
pip install pillow
```

## 快速开始

### 1. 基础训练

```python
from frame_prediction_rssm.pipeline import FramePredictionPipeline
from frame_prediction_rssm.training import create_train_state, train_step

# 初始化
rng = jax.random.PRNGKey(42)
train_state, apply_fn = create_train_state(rng, learning_rate=1e-4)

# 训练步骤
frames_1to4 = jnp.zeros((batch_size, 4, 64, 64, 3), dtype=jnp.uint8)
target_frame_5 = jnp.zeros((batch_size, 64, 64, 3), dtype=jnp.uint8)

train_state, metrics = train_step(
    train_state,
    frames_1to4,
    target_frame_5,
    apply_fn
)
```

### 2. 推理

```python
from frame_prediction_rssm.training import predict

predicted_frame, point_cloud = predict(
    train_state,
    frames_1to4,
    apply_fn
)

# 访问点云数据
positions = point_cloud['positions']      # [batch, num_points, 3]
features = point_cloud['features']         # [batch, num_points, feature_dim]
confidences = point_cloud['confidences']   # [batch, num_points]
```

### 3. 使用配置预设

```python
from frame_prediction_rssm.config import get_config

# 获取预设配置
config = get_config('small')  # 'tiny', 'small', 'medium', 'large'

# 转换为字典用于初始化
train_state, apply_fn = create_train_state(rng, **config.to_dict())
```

## 示例脚本

### 训练示例

```bash
cd examples
python train_frame_prediction.py
```

这将：
- 生成虚拟数据
- 初始化模型
- 运行训练循环
- 评估模型
- 保存预测结果

### 推理示例

```bash
cd examples
python inference.py --frame_dir ./frames --output_dir ./outputs --model_size small
```

参数：
- `--frame_dir`: 包含输入帧的目录 (frame_0.png, frame_1.png, ...)
- `--output_dir`: 输出目录 (默认: ./outputs)
- `--model_size`: 模型大小 (tiny, small, medium)

输出文件：
- `predicted_frame.npy` / `predicted_frame.png`: 预测的第5帧
- `point_cloud_positions.npy`: 点云XYZ坐标
- `point_cloud_features.npy`: 稠密特征向量
- `point_cloud_confidences.npy`: 置信度分数
- `predictions.ply`: PLY格式的可视化点云

## 架构详解

### RSSM隐空间预测

关键思想：

1. **确定性状态** (`deter`): 无损信息压缩，维度较大 (256-512)
2. **随机状态** (`stoch`): 离散隐变量 (stoch_size × classes)
3. **动态模型**: 预测 $s_{t+1} = f(s_t, a_t, o_t)$

```
Frame t → Encoder → Tokens
                ↓
         RSSM Core (融合action)
                ↓
         Stochastic Encoder
                ↓
         S_{t+1} = (deter_{t+1}, stoch_{t+1})
```

### 4D点云生成

从隐空间生成稠密特征的4D点云：

```
RSSM State (deter, stoch)
    ↓
Dense Feature Extractor (H×W×F)
    ↓
┌─────────────────────────┐
│ XY Grid + Depth Pred    │ → Positions [B, H×W, 3]
│ Feature Refinement      │ → Features [B, H×W, F]
│ Confidence Prediction   │ → Confidences [B, H×W]
└─────────────────────────┘
```

## 模型配置

### 微调尺寸

```python
PipelineConfig(
    encoder_hidden_dim=64,        # 编码器隐层大小
    encoder_latent_dim=256,       # 编码器输出维度
    encoder_conv_layers=4,        # 卷积层数
    encoder_temporal_layers=2,    # 时间编码层数
    
    rssm_deter_size=256,          # 确定性状态维度
    rssm_stoch_size=32,           # 随机状态维度
    rssm_classes=32,              # 每个随机变量的类数
    
    decoder_hidden_dim=64,        # 解码器隐层大小
    
    pc_feature_dim=256,           # 点云特征维度
    pc_spatial_height=16,         # 点云空间分辨率
    pc_spatial_width=16,
    pc_depth_range=1.0,           # 深度范围
)
```

## 损失函数

```python
Total Loss = Frame_Loss + lambda_pc * PC_Loss

Frame_Loss = MSE(predicted_frame, target_frame)

PC_Loss = lambda_confidence * Entropy(confidences) 
        + lambda_depth * Smoothness(depth_map)
```

## 性能

在标准配置下 (小模型)：
- 参数量: ~2.5M
- 推理时间: ~100ms (批大小=1)
- 点云大小: 256点 (16×16)

## 扩展建议

1. **改进动态模型**
   - 添加自回归预测多步
   - 条件变分编码器 (CVAE) 用于观测

2. **增强点云**
   - 支持可变分辨率点云
   - 法线向量预测
   - 语义分割

3. **多任务学习**
   - 光流预测
   - 深度估计
   - 动作预测

4. **高效性改进**
   - 模型量化
   - 知识蒸馏
   - 动态批处理

## 参考

基于以下工作：
- **Dreamer V3**: Danijar Hafner et al., "Mastering Diverse Domains through World Models"
- **RSSM**: 递归状态空间模型用于视频预测和规划

## License

与Dreamer V3相同
