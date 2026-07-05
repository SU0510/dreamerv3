#!/usr/bin/env python3
"""
快速启动脚本 - 5分钟内体验帧预测和4D点云生成
"""

import sys
from pathlib import Path


def _to_reference_point_cloud(image, numpy):
    height, width = image.shape[:2]
    image_f = image.astype(numpy.float32) / 255.0
    gray = image_f.mean(axis=-1)
    y_coords = numpy.linspace(-1.0, 1.0, height)
    x_coords = numpy.linspace(-1.0, 1.0, width)
    yy, xx = numpy.meshgrid(y_coords, x_coords, indexing="ij")
    positions = numpy.stack([xx, yy, gray], axis=-1).reshape(-1, 3)
    features = image_f.reshape(-1, 3)
    confidences = numpy.ones((height * width,), dtype=numpy.float32)
    return {
        "positions": positions[None, ...],
        "features": features[None, ...],
        "confidences": confidences[None, ...],
        "depth_map": gray[None, ...],
    }


def _visualize_results(frames_1to4, target_frame_5, predicted_frame, point_cloud, output_dir, numpy):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装，跳过可视化。")
        return

    reference_pc = _to_reference_point_cloud(target_frame_5, numpy)

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 5)

    for i in range(4):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frames_1to4[0, i])
        ax.set_title(f"Input {i + 1}")
        ax.axis("off")

    ax = fig.add_subplot(gs[0, 4])
    ax.imshow(target_frame_5)
    ax.set_title("Ground Truth Frame 5")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(predicted_frame[0])
    ax.set_title("Predicted Frame 5")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(target_frame_5)
    ax.set_title("GT Frame 5")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 2])
    ax.imshow(reference_pc["depth_map"][0], cmap="viridis")
    ax.set_title("GT Pseudo Point Cloud Depth")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 3])
    ax.imshow(point_cloud["depth_map"], cmap="viridis")
    ax.set_title("Predicted Point Cloud Depth")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 4], projection="3d")
    ref_positions = reference_pc["positions"][0]
    pred_positions = point_cloud["positions"][0]
    ref_conf = reference_pc["confidences"][0]
    pred_conf = point_cloud["confidences"][0]

    ax.scatter(ref_positions[:, 0], ref_positions[:, 1], ref_positions[:, 2], c=ref_conf, cmap="Blues", s=4, alpha=0.3, label="GT")
    ax.scatter(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], c=pred_conf, cmap="Reds", s=4, alpha=0.3, label="Pred")
    ax.set_title("GT vs Pred 3D Points")
    ax.legend(loc="upper right")
    ax.view_init(elev=20, azim=40)

    plt.tight_layout()
    fig_path = output_dir / "comparison_visualization.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"可视化已保存到 {fig_path}")

def quick_start():
    """快速启动演示"""
    
    print("\n" + "="*60)
    print("Frame Prediction RSSM - Quick Start")
    print("="*60)
    
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        print("\n✓ JAX 已安装")
    except ImportError:
        print("\n✗ JAX 未安装，请运行: pip install jax jaxlib flax optax numpy")
        return False
    
    try:
        from frame_prediction_rssm.pipeline import FramePredictionPipeline
        from frame_prediction_rssm.training import create_train_state, predict
        from frame_prediction_rssm.config import get_config
        print("✓ frame_prediction_rssm 已安装")
    except ImportError as e:
        print(f"✗ frame_prediction_rssm 导入失败: {e}")
        print("  请确保在 frame_prediction_rssm 目录中")
        return False
    
    # 创建输出目录
    output_dir = Path("quick_start_outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "-"*60)
    print("演示 1: 初始化模型")
    print("-"*60)
    
    # 初始化
    rng = jax.random.PRNGKey(42)
    config = get_config('small')
    print(f"使用 'small' 模型预设")
    print(f"  - 编码器隐层: {config.encoder_hidden_dim}")
    print(f"  - RSSM确定性状态: {config.rssm_deter_size}")
    print(f"  - 随机状态: {config.rssm_stoch_size}")
    
    print("\n初始化模型...", end='', flush=True)
    train_state, apply_fn = create_train_state(rng, **config.to_dict())
    print(" ✓")
    
    # 计算参数数量
    def count_params(params):
        return sum(p.size for p in jax.tree.leaves(params))
    
    n_params = count_params(train_state.params)
    print(f"模型参数: {n_params:,}")
    
    print("\n" + "-"*60)
    print("演示 2: 生成虚拟帧序列")
    print("-"*60)
    
    # 生成虚拟数据
    batch_size = 2
    frames_1to4 = np.random.randint(0, 255, (batch_size, 4, 64, 64, 3), dtype=np.uint8)
    target_frame_5 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    print(f"生成 {batch_size} 个批次的4帧序列")
    print(f"  形状: {frames_1to4.shape}")
    
    print("\n" + "-"*60)
    print("演示 3: 推理 - 预测第5帧和点云")
    print("-"*60)
    
    print("\n正在进行推理...", end='', flush=True)
    predicted_frame, point_cloud = predict(train_state, frames_1to4, apply_fn)
    print(" ✓")
    
    print("\n预测结果:")
    print(f"  预测帧形状: {predicted_frame.shape}")
    print(f"  点云位置: {point_cloud['positions'].shape}")
    print(f"  点云特征: {point_cloud['features'].shape}")
    print(f"  置信度: {point_cloud['confidences'].shape}")
    
    print("\n点云统计:")
    for b in range(batch_size):
        conf = point_cloud['confidences'][b]
        depth = point_cloud['depth_map'][b]
        valid = (conf > 0.3).sum()
        print(f"  批次 {b+1}:")
        print(f"    - 有效点 (conf>0.3): {valid}/{len(conf)}")
        print(f"    - 平均置信度: {conf.mean():.4f}")
        print(f"    - 深度范围: [{depth.min():.4f}, {depth.max():.4f}]")
    
    print("\n" + "-"*60)
    print("演示 4: 保存输出")
    print("-"*60)
    
    # 保存结果
    np.save(output_dir / "predicted_frame.npy", predicted_frame)
    np.save(output_dir / "point_cloud_positions.npy", point_cloud['positions'])
    np.save(output_dir / "point_cloud_features.npy", point_cloud['features'])
    np.save(output_dir / "point_cloud_confidences.npy", point_cloud['confidences'])
    np.save(output_dir / "target_frame_5.npy", target_frame_5)
    
    print(f"✓ 结果已保存到 {output_dir}/")
    print(f"  - predicted_frame.npy")
    print(f"  - point_cloud_positions.npy")
    print(f"  - point_cloud_features.npy")
    print(f"  - point_cloud_confidences.npy")
    print(f"  - target_frame_5.npy")
    
    print("\n" + "-"*60)
    print("演示 5: 可视化点云数据")
    print("-"*60)
    
    positions = point_cloud['positions'][0]  # 第一个批次
    features = point_cloud['features'][0]
    confidences = point_cloud['confidences'][0]
    
    # 过滤低置信度的点
    valid_mask = confidences > 0.3
    n_valid = valid_mask.sum()
    
    print(f"\n点云数据 (批次 0):")
    print(f"  总点数: {len(positions)}")
    print(f"  有效点 (conf>0.3): {n_valid}")
    print(f"  过滤比例: {n_valid/len(positions)*100:.1f}%")
    
    # 显示一些点的坐标
    print(f"\n前5个点的坐标 (XYZ):")
    for i in range(min(5, len(positions))):
        x, y, z = positions[i]
        conf = confidences[i]
        print(f"    点{i+1}: X={x:7.4f} Y={y:7.4f} Z={z:7.4f} (置信度: {conf:.4f})")

    print("\n" + "-"*60)
    print("演示 6: 可视化对比")
    print("-"*60)
    print("显示: 输入前4帧、原始第5帧、预测第5帧、两套点云和3D叠加图")
    _visualize_results(frames_1to4, target_frame_5, predicted_frame, point_cloud, output_dir, np)
    
    print("\n" + "="*60)
    print("✓ 快速启动演示完成！")
    print("="*60)
    
    print("\n接下来可以尝试:")
    print("  1. 修改模型大小: get_config('tiny'|'medium'|'large')")
    print("  2. 运行训练: python examples/train_frame_prediction.py")
    print("  3. 使用自己的数据: python examples/inference.py --frame_dir <dir>")
    print("  4. 详见: README_FramePrediction.md")
    
    return True


def provide_examples():
    """提供使用示例代码"""
    
    print("\n" + "="*60)
    print("使用示例代码片段")
    print("="*60)
    
    examples = {
        "基础推理": """
from frame_prediction_rssm.training import create_train_state, predict
import jax, jax.numpy as jnp

rng = jax.random.PRNGKey(42)
state, apply_fn = create_train_state(rng)

frames = jnp.zeros((1, 4, 64, 64, 3), dtype=jnp.uint8)
frame_pred, pc = predict(state, frames, apply_fn)
""",
        
        "使用不同模型大小": """
from frame_prediction_rssm.config import get_config

for size in ['tiny', 'small', 'medium']:
    config = get_config(size)
    state, apply_fn = create_train_state(rng, **config.to_dict())
""",
        
        "训练": """
from frame_prediction_rssm.training import train_step

for epoch in range(10):
    state, metrics = train_step(
        state, 
        frames_batch,     # [B, 4, H, W, 3]
        target_frame_5,   # [B, H, W, 3]
        apply_fn
    )
    print(f"Loss: {metrics['total_loss']:.6f}")
""",
        
        "访问点云": """
positions = point_cloud['positions']       # [B, num_points, 3]
features = point_cloud['features']         # [B, num_points, 256]
confidences = point_cloud['confidences']   # [B, num_points]

# 过滤低置信度点
valid_mask = confidences > 0.3
valid_positions = positions[valid_mask]
""",
    }
    
    for title, code in examples.items():
        print(f"\n{title}:")
        print(code)


if __name__ == "__main__":
    success = quick_start()
    
    if success and "--examples" in sys.argv:
        provide_examples()
    
    sys.exit(0 if success else 1)
