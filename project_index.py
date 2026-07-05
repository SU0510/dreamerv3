"""
Frame Prediction RSSM - 项目完整索引

该项目实现了基于 Dreamer V3 思想的帧预测系统，
从前4帧预测第5帧并生成稠密特征的4D点云。
"""

PROJECT_STRUCTURE = """
dreamerv3/
│
├── frame_prediction_rssm/              ← 核心项目目录
│   ├── __init__.py                     # 包初始化
│   ├── rssm_core.py                    # RSSM 核心模块
│   │   - RSSMState: 状态表示 (deter + stoch)
│   │   - RSSMCore: 确定性状态更新
│   │   - StochasticEncoder: 隐空间编码
│   │   - RSSM: 单步预测
│   │   - RSSMSequence: 序列处理 (4步)
│   │
│   ├── encoder.py                      # 图像编码器
│   │   - FrameEncoder: CNN视觉编码
│   │   - TemporalEncoder: 时间注意力
│   │   - MultiFrameEncoder: 组合编码
│   │
│   ├── decoder.py                      # 解码+点云生成
│   │   - FeatureDecoder: 帧重建
│   │   - DenseFeatureExtractor: 特征提取
│   │   - PointCloudGenerator: 点云生成
│   │   - AdaptivePointCloudGenerator: 自适应点云
│   │
│   ├── pipeline.py                     # 完整管道
│   │   - FramePredictionPipeline: 端到端处理
│   │   - InferenceModel: 推理包装器
│   │
│   ├── training.py                     # 训练工具
│   │   - create_train_state: 初始化
│   │   - train_step: JIT训练步骤
│   │   - eval_step: 评估步骤
│   │   - predict: 推理接口
│   │
│   ├── config.py                       # 配置预设
│   │   - PipelineConfig: 配置数据类
│   │   - TrainingConfig: 训练超参数
│   │   - CONFIGS: 预定义模型 (tiny/small/medium/large)
│   │
│   ├── requirements.txt                # Python依赖
│   └── README.md                       # 详细技术文档
│
├── examples/                           ← 示例脚本
│   ├── README.md                       # 示例说明
│   ├── train_frame_prediction.py       # 训练演示脚本
│   ├── inference.py                    # 推理脚本 (CLI支持)
│   └── notebook_example.py             # Jupyter示例
│
├── quick_start.py                      # 快速启动脚本 (5分钟体验)
├── README_FramePrediction.md           # 项目总览文档
└── [其他 Dreamer V3 文件...]
"""

QUICK_START_GUIDE = """
========================================
快速开始 - 3步上手
========================================

1. 安装依赖
   cd frame_prediction_rssm
   pip install -r requirements.txt

2. 运行快速启动
   python quick_start.py

3. 尝试推理
   python examples/inference.py --model_size small

输出位置: outputs/ 或 quick_start_outputs/
"""

CORE_CONCEPTS = """
========================================
核心概念
========================================

1. RSSM (Recurrent State Space Model)
   ├─ 确定性状态 (deter): 256-512维，记录所有信息
   ├─ 随机状态 (stoch): 离散潜变量，维度 [stoch_size, classes]  
   └─ 预测: s_{t+1} = RSSM(s_t, action_t)

2. 帧预测管道
   └─ Frames 1-4 → Encoder → RSSM (4步) → Decoder → Frame 5

3. 4D点云生成
   └─ RSSM State → DenseFeatureExtractor → [Positions + Features + Confidence]
      └─ Positions: XYZ坐标 (16×16=256个点)
      └─ Features: 256维特征向量
      └─ Confidence: 0-1置信度分数

4. 损失函数
   └─ Total = L_frame + λ·L_pc
      ├─ L_frame = MSE(predicted, target)
      └─ L_pc = entropy(confidence) + smoothness(depth)
"""

FILE_PURPOSES = """
========================================
文件功能速览
========================================

RSSM 核心:
  rssm_core.py      → 状态表示 + 动态模型 + 预测

编码:
  encoder.py        → 帧特征提取 + 时间编码

解码:
  decoder.py        → 帧重建 + 点云生成

集成:
  pipeline.py       → 端到端处理 (Encoder → RSSM → Decoder)

训练:
  training.py       → 初始化 + 训练循环 + 推理
  config.py         → 配置管理 (tiny/small/medium/large)

示例:
  quick_start.py    → 5分钟快速体验
  examples/train_... → 完整训练演示
  examples/infer... → 推理脚本 (支持CLI和API)
  
文档:
  README.md         → 技术细节
  README_FramePred. → 项目总览
"""

API_EXAMPLES = """
========================================
API 使用示例
========================================

### 基础推理
from frame_prediction_rssm.training import create_train_state, predict
import jax

rng = jax.random.PRNGKey(42)
state, apply_fn = create_train_state(rng)

# 前4帧 [batch, 4, height, width, 3]
predicted_frame, point_cloud = predict(state, frames_1to4, apply_fn)

### 访问点云
pc = point_cloud
positions = pc['positions']        # [B, 256, 3] 
features = pc['features']          # [B, 256, 256]
confidences = pc['confidences']    # [B, 256]

# 过滤低质量点
valid = confidences > 0.3
valid_pos = positions[valid]

### 使用预设配置
from frame_prediction_rssm.config import get_config

config = get_config('small')  # 'tiny', 'small', 'medium', 'large'
state, apply_fn = create_train_state(rng, **config.to_dict())

### 训练
from frame_prediction_rssm.training import train_step

for epoch in range(10):
    state, metrics = train_step(
        state, frames, target, apply_fn
    )
    print(metrics['total_loss'])
"""

MODEL_SIZES = """
========================================
模型大小对比
========================================

        参数量    推理时间    质量
tiny      250K      50ms     ◆
small   2.5M       100ms     ◆◆◆ (推荐)
medium   12M       200ms     ◆◆◆◆
large    60M       500ms     ◆◆◆◆◆

推荐: small (快速 + 高质量)
"""

TROUBLESHOOTING = """
========================================
常见问题
========================================

Q: ImportError: No module named 'jax'
A: pip install jax jaxlib

Q: 推理速度慢
A: 使用更小的模型 get_config('tiny')
   或减少点云分辨率 pc_spatial_height=8

Q: OOM 内存不足
A: 减小batch_size
   使用 config.py 中的 'tiny' 模型

Q: 预测效果不好
A: 1. 增加训练轮数
   2. 调整超参数 (learning_rate, lambda_*)
   3. 使用更大的模型 (medium/large)
   4. 增加真实训练数据

Q: 点云置信度都很低
A: 1. 检查你的目标帧与输入帧的相关性
   2. 调整 lambda_confidence 的权重
   3. 增加训练步数
"""

FILE_TREE = """
========================================
完整文件树
========================================

frame_prediction_rssm/
├── __init__.py                           (170字节)
├── rssm_core.py                          (8.5KB)   [RSSM核心]
├── encoder.py                            (7.2KB)   [编码器]
├── decoder.py                            (10.1KB)  [解码+点云]
├── pipeline.py                           (6.8KB)   [管道]
├── training.py                           (8.3KB)   [训练工具]
├── config.py                             (3.2KB)   [配置]
├── requirements.txt                      (89字节)  [依赖]
└── README.md                             (12.4KB)  [技术文档]

examples/
├── README.md                             (文件说明)
├── train_frame_prediction.py             (9.5KB)   [训练演示]
├── inference.py                          (11.2KB)  [推理脚本]
└── notebook_example.py                   (示例代码)

根目录/
├── quick_start.py                        (快速启动)
├── README_FramePrediction.md             (项目总览)
└── [Dreamer V3其他文件...]

总代码量: ~65KB 核心实现
"""

if __name__ == "__main__":
    import sys
    
    content_map = {
        "structure": PROJECT_STRUCTURE,
        "quick": QUICK_START_GUIDE,
        "concepts": CORE_CONCEPTS,
        "files": FILE_PURPOSES,
        "api": API_EXAMPLES,
        "models": MODEL_SIZES,
        "help": TROUBLESHOOTING,
        "tree": FILE_TREE,
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in content_map:
        print(content_map[sys.argv[1]])
    else:
        print("\n" + "="*60)
        print("Frame Prediction RSSM - 项目索引")
        print("="*60)
        print("\n使用方法: python project_index.py <section>")
        print("\n可用章节:")
        for key in content_map.keys():
            print(f"  - {key}")
        print("\n例如: python project_index.py quick")
        print("\n显示所有信息:")
        for key, content in content_map.items():
            print(content)
            print()
