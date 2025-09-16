import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import match_anchors_to_targets

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3, obj_w=1.0, cls_w=1.0, loc_w=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.obj_w = obj_w
        self.cls_w = cls_w
        self.loc_w = loc_w
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, predictions, targets, anchors_per_scale):
        """
        predictions: list of [B, A*(5+C), H, W]
        targets: list of dict{'boxes','labels'} 长度 B
        anchors_per_scale: list of [A,4]
        """
        B = predictions[0].shape[0]
        loss_obj_total, loss_cls_total, loss_loc_total = 0.0, 0.0, 0.0

        for p, anchors in zip(predictions, anchors_per_scale):
            B, _, H, W = p.shape
            A = anchors.shape[0] // (H*W)
            p = p.view(B, A, 5 + self.num_classes, H, W).permute(0,1,3,4,2)  # [B,A,H,W,5+C]
            # 预测拆分
            pred_box = p[..., 0:4]       # tx,ty,tw,th（这里直接当回归量）
            pred_obj = p[..., 4]         # objectness logit
            pred_cls = p[..., 5:]        # class logits
            pred_box = pred_box.reshape(B, -1, 4)
            pred_obj = pred_obj.reshape(B, -1)
            pred_cls = pred_cls.reshape(B, -1, self.num_classes)

            anchors = anchors.to(pred_obj.device)

            for b in range(B):
                t = targets[b]
                mlabel, mbox, pos_mask, neg_mask = match_anchors_to_targets(
                    anchors, t["boxes"].to(anchors.device), t["labels"].to(anchors.device)
                )
                # objectness: pos=1 neg=0 (忽略中间区间)
                obj_target = torch.zeros_like(pred_obj[b])
                obj_target[pos_mask] = 1.0
                obj_loss = self.bce(pred_obj[b], obj_target)
                # hard negative mining 3:1
                num_pos = max(pos_mask.sum().item(), 1)
                neg_k = min(neg_mask.sum().item(), 3 * num_pos)
                if neg_k > 0:
                    neg_scores = obj_loss[neg_mask]
                    topk = torch.topk(neg_scores, k=neg_k).indices
                    selected = torch.zeros_like(neg_mask)
                    selected[neg_mask] = False
                    # 用 scatter 选择
                    neg_indices = torch.where(neg_mask)[0][topk]
                    selected[neg_indices] = True
                    obj_keep = pos_mask | selected
                else:
                    obj_keep = pos_mask
                loss_obj_total += obj_loss[obj_keep].mean()

                # 分类（仅正样本）
                if pos_mask.any():
                    cls_target = (mlabel[pos_mask] - 1).clamp(min=0)  # 映射到 0..C-1
                    cls_loss = self.ce(pred_cls[b][pos_mask], cls_target)
                    loss_cls_total += cls_loss.mean()
                    # 回归（Smooth L1）
                    loc_loss = F.smooth_l1_loss(pred_box[b][pos_mask], mbox[pos_mask], reduction="mean")
                    loss_loc_total += loc_loss
                else:
                    loss_cls_total += torch.tensor(0.0, device=pred_obj.device)
                    loss_loc_total += torch.tensor(0.0, device=pred_obj.device)

        loss = {
            "loss_obj": loss_obj_total / len(predictions),
            "loss_cls": loss_cls_total / len(predictions),
            "loss_loc": loss_loc_total / len(predictions),
        }
        loss["loss_total"] = self.obj_w*loss["loss_obj"] + self.cls_w*loss["loss_cls"] + self.loc_w*loss["loss_loc"]
        return loss