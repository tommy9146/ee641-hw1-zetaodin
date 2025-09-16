import torch

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    """
    仅使用 1:1 比例；每个 cell 的中心放多种平方框（以边长表示）
    返回每个尺度一个 [N,4] 张量 (x1,y1,x2,y2)
    """
    anchors_all = []
    for (H, W), scales in zip(feature_map_sizes, anchor_scales):
        stride_y = image_size / H
        stride_x = image_size / W
        ys = torch.arange(H) + 0.5
        xs = torch.arange(W) + 0.5
        cy, cx = torch.meshgrid(ys, xs, indexing="ij")
        cx = cx * stride_x
        cy = cy * stride_y
        anchors_scale = []
        for s in scales:
            w = torch.full_like(cx, float(s))
            h = torch.full_like(cy, float(s))
            x1 = cx - w/2
            y1 = cy - h/2
            x2 = cx + w/2
            y2 = cy + h/2
            anchors_scale.append(torch.stack([x1,y1,x2,y2], dim=-1).reshape(-1,4))
        anchors_all.append(torch.cat(anchors_scale, dim=0))  # [H*W*len(scales),4]
    return anchors_all

def compute_iou(boxes1, boxes2):
    # boxes1: [N,4], boxes2: [M,4]
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    x1 = torch.max(boxes1[:,None,0], boxes2[None,:,0])
    y1 = torch.max(boxes1[:,None,1], boxes2[None,:,1])
    x2 = torch.min(boxes1[:,None,2], boxes2[None,:,2])
    y2 = torch.min(boxes1[:,None,3], boxes2[None,:,3])
    inter = (x2-x1).clamp(min=0) * (y2-y1).clamp(min=0)
    area1 = (boxes1[:,2]-boxes1[:,0]).clamp(min=0) * (boxes1[:,3]-boxes1[:,1]).clamp(min=0)
    area2 = (boxes2[:,2]-boxes2[:,0]).clamp(min=0) * (boxes2[:,3]-boxes2[:,1]).clamp(min=0)
    union = area1[:,None] + area2[None,:] - inter
    iou = inter / (union + 1e-6)
    return iou

def match_anchors_to_targets(anchors, target_boxes, target_labels,
                             pos_threshold=0.5, neg_threshold=0.3):
    """
    简化版匹配：最佳匹配 + 阈值
    返回：
      matched_labels: [A] (0 背景, 1..C)
      matched_boxes:  [A,4]
      pos_mask, neg_mask: bool
    """
    A = anchors.shape[0]
    matched_labels = torch.zeros((A,), dtype=torch.long)
    matched_boxes = torch.zeros((A,4), dtype=torch.float32)
    if target_boxes.numel() == 0:
        neg_mask = torch.ones(A, dtype=torch.bool)
        pos_mask = torch.zeros(A, dtype=torch.bool)
        return matched_labels, matched_boxes, pos_mask, neg_mask

    iou = compute_iou(anchors, target_boxes)  # [A,T]
    max_iou, max_idx = iou.max(dim=1)

    # 正负样本
    pos_mask = max_iou >= pos_threshold
    neg_mask = max_iou < neg_threshold
    # 为每个 GT 至少分配一个最佳 anchor（保证召回）
    gt_best_iou, gt_best_anchor = iou.max(dim=0)
    pos_mask[gt_best_anchor] = True

    matched_boxes[pos_mask] = target_boxes[max_idx[pos_mask]]
    # 类别从 1 开始编码（0 预留给背景）
    tmp = torch.zeros_like(matched_labels)
    tmp[pos_mask] = target_labels[max_idx[pos_mask]] + 1
    matched_labels = tmp
    return matched_labels, matched_boxes, pos_mask, neg_mask