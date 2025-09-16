import os, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet

def extract_keypoints_from_heatmaps(heatmaps):
    # heatmaps: [B,K,H,W] -> coords [B,K,2] in [0,1] (x,y)
    B,K,H,W = heatmaps.shape
    hm = heatmaps.reshape(B, K, -1)
    idx = hm.argmax(-1)                         # [B,K]
    y = (idx // W).float() / H
    x = (idx %  W).float() / W
    return torch.stack([x,y], dim=-1)

def compute_pck(pred, gt, thr=0.2, norm_by="bbox"):
    """
    pred, gt: [N,K,2] (in [0,1] range)
    normalize by image diagonal (since是合成数据，等价)
    """
    N,K,_ = pred.shape
    # 转像素距离，以 128 对应
    pd = pred.clone()*128.0
    gd = gt.clone()*128.0
    d = torch.linalg.norm(pd - gd, dim=-1)     # [N,K]
    diag = (2*(128**2))**0.5                   # 对角线
    thresh = thr*diag
    acc = (d <= thresh).float().mean().item()
    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 准备数据
    root=".."; data_root=os.path.join(root,"datasets","keypoints")
    val_ds_hm = KeypointDataset(os.path.join(data_root,"val"),
                                os.path.join(data_root,"val_annotations.json"),
                                output_type="heatmap", heatmap_size=64, sigma=2.0)
    val_ds_rg = KeypointDataset(os.path.join(data_root,"val"),
                                os.path.join(data_root,"val_annotations.json"),
                                output_type="regression")
    loader_hm = torch.utils.data.DataLoader(val_ds_hm, batch_size=64, shuffle=False)
    loader_rg = torch.utils.data.DataLoader(val_ds_rg, batch_size=64, shuffle=False)

    # 载入模型
    hm = HeatmapNet(5).to(device); hm.load_state_dict(torch.load("results/heatmap_model.pth", map_location=device)); hm.eval()
    rg = RegressionNet(5).to(device); rg.load_state_dict(torch.load("results/regression_model.pth", map_location=device)); rg.eval()

    preds_h, gts = [], []
    with torch.no_grad():
        for (x_h, _), (x_r, y_r) in zip(loader_hm, loader_rg):
            x_h=x_h.to(device); x_r=x_r.to(device)
            ph = hm(x_h)                           # [B,5,64,64]
            ph = extract_keypoints_from_heatmaps(ph).cpu()   # [B,5,2] in [0,1]
            # 回归的 GT：y_r 已经是 [B,10] in [0,1]
            g = y_r.view(-1,5,2)
            preds_h.append(ph); gts.append(g)
    preds_h = torch.cat(preds_h,0); gts = torch.cat(gts,0)

    # 评估 Heatmap
    pck_vals = {t: compute_pck(preds_h, gts, thr=t) for t in [0.05,0.1,0.15,0.2]}
    print("Heatmap PCK:", pck_vals)

    # 评估 Regression
    preds_r = []
    with torch.no_grad():
        for x_r, y_r in loader_rg:
            pr = rg(x_r.to(device)).cpu().view(-1,5,2)
            preds_r.append(pr)
    preds_r = torch.cat(preds_r,0)
    pck_vals_r = {t: compute_pck(preds_r, gts, thr=t) for t in [0.05,0.1,0.15,0.2]}
    print("Regression PCK:", pck_vals_r)

    # 画 PCK 曲线
    os.makedirs("results/visualizations", exist_ok=True)
    ts = [0.05,0.1,0.15,0.2]
    plt.plot(ts, [pck_vals[t] for t in ts], marker='o', label='Heatmap')
    plt.plot(ts, [pck_vals_r[t] for t in ts], marker='o', label='Regression')
    plt.xlabel("PCK Threshold"); plt.ylabel("Accuracy"); plt.legend()
    plt.savefig("results/visualizations/pck_curve.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()