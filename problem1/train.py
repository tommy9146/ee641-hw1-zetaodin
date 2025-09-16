import os, json, time
import torch
from torch.utils.data import DataLoader

from dataset import ShapeDetectionDataset, collate_fn
from model import MultiScaleDetector
from loss import DetectionLoss
from utils import generate_anchors

def ensure_dirs():
    os.makedirs("results/visualizations", exist_ok=True)

def get_datasets(root=".."):
    data_root = os.path.join(root, "datasets", "detection")
    img_tr = os.path.join(data_root, "train")
    ann_tr = os.path.join(data_root, "train_annotations.json")
    img_va = os.path.join(data_root, "val")
    ann_va = os.path.join(data_root, "val_annotations.json")
    train_ds = ShapeDetectionDataset(img_tr, ann_tr)
    val_ds   = ShapeDetectionDataset(img_va, ann_va)
    return train_ds, val_ds

def build_anchors(device):
    feature_map_sizes = [(56,56), (28,28), (14,14)]
    anchor_scales = [[16,24,32], [48,64,96], [96,128,192]]
    anchors = generate_anchors(feature_map_sizes, anchor_scales, image_size=224)
    anchors = [a.to(device) for a in anchors]
    return anchors

def train_one_epoch(model, loader, criterion, optimizer, device, anchors):
    model.train()
    running = {"loss_total":0.0, "loss_obj":0.0, "loss_cls":0.0, "loss_loc":0.0}
    for imgs, targets in loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss_dict = criterion(preds, targets, anchors)
        loss_dict["loss_total"].backward()
        optimizer.step()
        for k in running:
            running[k] += loss_dict[k].item()
    for k in running:
        running[k] /= max(len(loader),1)
    return running

@torch.no_grad()
def validate(model, loader, criterion, device, anchors):
    model.eval()
    running = {"loss_total":0.0, "loss_obj":0.0, "loss_cls":0.0, "loss_loc":0.0}
    for imgs, targets in loader:
        imgs = imgs.to(device)
        preds = model(imgs)
        loss_dict = criterion(preds, targets, anchors)
        for k in running:
            running[k] += loss_dict[k].item()
    for k in running:
        running[k] /= max(len(loader),1)
    return running

def main():
    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = get_datasets(root="..")
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    criterion = DetectionLoss(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    anchors = build_anchors(device)

    num_epochs = 2  # 先小跑 1-2 轮验证流程；正式训练再改为 50
    best_val = float("inf")
    log = {"train":[], "val":[]}

    for epoch in range(1, num_epochs+1):
        t0 = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device, anchors)
        va = validate(model, val_loader, criterion, device, anchors)
        t1 = time.time()

        log["train"].append(tr); log["val"].append(va)
        with open("results/training_log.json", "w") as f:
            json.dump(log, f, indent=2)

        print(f"[Epoch {epoch}] "
              f"train_total={tr['loss_total']:.4f}  val_total={va['loss_total']:.4f}  "
              f"time={t1-t0:.1f}s")

        if va["loss_total"] < best_val:
            best_val = va["loss_total"]
            torch.save(model.state_dict(), "results/best_model.pth")

if __name__ == "__main__":
    main()