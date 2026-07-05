"""
Jupyter示例notebook - 交互式演示
"""

# 为适应Jupyter环境，这是一个脚本版本的notebook
# 可以通过以下方式在notebook中使用：

def notebook_example():
    """
    在Jupyter notebook中运行此代码:
    
    ```python
    import jax
    import jax.numpy as jnp
    import numpy as np
    from pathlib import Path
    
    # 导入
    from frame_prediction_rssm.pipeline import FramePredictionPipeline
    from frame_prediction_rssm.training import create_train_state, predict
    from frame_prediction_rssm.config import get_config
    
    # 1. 初始化
    rng = jax.random.PRNGKey(42)
    config = get_config('small')
    train_state, apply_fn = create_train_state(rng, **config.to_dict())
    
    # 2. 加载或生成帧
    frames_1to4 = np.random.randint(0, 255, (1, 4, 64, 64, 3), dtype=np.uint8)
    
    # 3. 预测
    predicted_frame, point_cloud = predict(train_state, frames_1to4, apply_fn)
    
    # 4. 可视化 (需要matplotlib)
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 显示输入帧
    axes[0].imshow(frames_1to4[0, -1])  # 最后一帧
    axes[0].set_title('Input Frame 4')
    axes[0].axis('off')
    
    # 显示预测帧
    axes[1].imshow(predicted_frame[0])
    axes[1].set_title('Predicted Frame 5')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 5. 点云统计
    print(f'Point Cloud Statistics:')
    print(f'  Positions: {point_cloud[\"positions\"].shape}')
    print(f'  Features: {point_cloud[\"features\"].shape}')
    print(f'  Avg Confidence: {point_cloud[\"confidences\"].mean():.4f}')
    print(f'  Valid Points (conf > 0.3): {(point_cloud[\"confidences\"] > 0.3).sum()}')
    
    # 6. 3D可视化 (需要plotly或mayavi)
    # 这里我们只展示如何提取信息
    positions = point_cloud['positions'][0]  # [H*W, 3]
    confidences = point_cloud['confidences'][0]  # [H*W]
    
    # 过滤低置信度的点
    valid_mask = confidences > 0.3
    valid_pos = positions[valid_mask]
    
    print(f'\\n{valid_pos.shape[0]} valid points after filtering')
    ```
    """
    print(__doc__)


if __name__ == '__main__':
    notebook_example()
