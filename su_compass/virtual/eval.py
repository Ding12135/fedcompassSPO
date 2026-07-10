"""
su_compass.virtual.eval — 虚拟实验全局模型评估工具。

在虚拟联邦学习主循环中，每次全局聚合完成后调用 evaluate_global_model()，
将结果写入 global_eval_trace.csv，用于绘制精度-时间曲线及创新方法对比。

评估策略：
    - 复用 client_0 的 APPFLClientAgent（含 val_dataloader 与 metric）
    - 将控制器持有的 global_model state_dict 加载到该 agent 的 trainer.model
    - 在验证集上跑一遍 forward，计算 loss 与 accuracy

这里只评估“当前全局模型质量”，不参与调度，也不改变虚拟时间。
因此 evaluate_global_model() 可以在每次聚合后同步调用，作为纯观测输出。

注意：
    - accuracy 返回 0–100 尺度，与 examples/metric/acc.py 一致
    - 标签需 reshape(-1) 再与 argmax 预测比较，避免 (N,1) vs (N,) 比较错误
"""

from typing import Tuple

import numpy as np
import torch


def evaluate_global_model(client_agent, global_model_state: dict) -> Tuple[float, float, int]:
    """在指定客户端的验证集上评估全局模型。

    Args:
        client_agent: 已配置 val_dataloader 的 APPFLClientAgent（通常为 client_0）。
        global_model_state: 控制器当前全局模型的 state_dict。

    Returns:
        (test_accuracy, test_loss, num_val_samples)：
            test_accuracy — 百分比精度（0–100）；
            test_loss     — 验证集平均交叉熵；
            num_val_samples — 参与评估的样本数。
    """
    trainer = client_agent.trainer
    device = trainer.train_configs.device

    # 加载最新全局权重并切换到 eval 模式
    trainer.model.load_state_dict(global_model_state)
    trainer.model.to(device)
    trainer.model.eval()

    if trainer.val_dataloader is None:
        return 0.0, 0.0, 0

    total_loss = 0.0
    target_true, target_pred = [], []
    num_batches = 0

    with torch.no_grad():
        for data, target in trainer.val_dataloader:
            data, target = data.to(device), target.to(device)
            output = trainer.model(data)
            loss = trainer.loss_fn(output, target)
            total_loss += float(loss.item())
            pred = output.argmax(dim=1, keepdim=True)
            # 展平为 (N,) 一维数组，确保与 metric/acc.py 的 == 比较正确
            target_true.append(target.cpu().numpy().reshape(-1))
            target_pred.append(pred.cpu().numpy().reshape(-1))
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0

    target_true = np.concatenate(target_true)
    target_pred = np.concatenate(target_pred)
    test_loss = total_loss / num_batches
    test_accuracy = float(trainer.metric(target_true, target_pred))
    return test_accuracy, test_loss, len(target_true)
