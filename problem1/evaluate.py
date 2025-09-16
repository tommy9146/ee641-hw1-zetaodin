import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    """Compute Average Precision for a single class."""
    # TODO: 按分数排序 -> IoU 匹配 -> 计算 Precision-Recall -> 积分算AP
    return 0.0

def visualize_detections(image, predictions, ground_truths, save_path):
    """Visualize predictions and ground truth boxes."""
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1,2,0).cpu().numpy(), cmap="gray")
    # GT green
    for b in ground_truths["boxes"]:
        x1,y1,x2,y2 = b.tolist()
        ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                       linewidth=2, edgecolor="g", facecolor="none"))
    # Pred red
    for row in predictions:
        x1,y1,x2,y2,score,label = row.tolist()
        ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                       linewidth=2, edgecolor="r", facecolor="none"))
    plt.savefig(save_path); plt.close()

def analyze_scale_performance(model, dataloader, anchors):
    """Analyze which scales detect which object sizes."""
    # TODO: 跑验证集 -> 统计小/中/大目标分别由哪个 scale 检出
    # 小: area < 32^2, 中: <96^2, 大: >=96^2
    return