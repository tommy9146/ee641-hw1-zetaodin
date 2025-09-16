import os, json, time
import torch
from torch.utils.data import DataLoader
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet

def get_loaders(root="..", mode="heatmap", batch_size=32, heatmap_size=64, sigma=2.0):
    data_root = os.path.join(root, "datasets", "keypoints")
    tr_img = os.path.join(data_root, "train")
    va_img = os.path.join(data_root, "val")
    tr_ann = os.path.join(data_root, "train_annotations.json")
    va_ann = os.path.join(data_root, "val_annotations.json")
    train_ds = KeypointDataset(tr_img, tr_ann, output_type=mode, heatmap_size=heatmap_size, sigma=sigma)
    val_ds   = KeypointDataset(va_img, va_ann, output_type=mode, heatmap_size=heatmap_size, sigma=sigma)
    loader_kw = dict(batch_size=batch_size, shuffle=(mode=="heatmap"), num_workers=2, pin_memory=True)
    return DataLoader(train_ds, **loader_kw), DataLoader(val_ds, **{**loader_kw, "shuffle": False})

def train_loop(model, train_loader, val_loader, criterion, optimizer, device, epochs, save_path):
    os.makedirs("results", exist_ok=True)
    log = {"train":[], "val":[]}
    best = 1e9
    for ep in range(1, epochs+1):
        model.train(); t0=time.time()
        tr_loss=0.0
        for x,y in train_loader:
            x=x.to(device); y=y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward(); optimizer.step()
            tr_loss += loss.item()
        tr_loss /= max(len(train_loader),1)

        # val
        model.eval(); va_loss=0.0
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(device); y=y.to(device)
                out=model(x)
                va_loss += criterion(out,y).item()
        va_loss /= max(len(val_loader),1)
        t1=time.time()
        log["train"].append(tr_loss); log["val"].append(va_loss)
        with open("results/training_log.json","w") as f: json.dump(log,f,indent=2)
        print(f"[Epoch {ep}] train={tr_loss:.4f}  val={va_loss:.4f}  time={t1-t0:.1f}s")
        if va_loss < best:
            best = va_loss
            torch.save(model.state_dict(), save_path)

def main():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Heatmap 模型
    tr, va = get_loaders(mode="heatmap", batch_size=32, heatmap_size=64, sigma=2.0)
    heatmap_model = HeatmapNet(num_keypoints=5).to(device)
    crit_h = torch.nn.MSELoss()
    opt_h = torch.optim.Adam(heatmap_model.parameters(), lr=1e-3)
    train_loop(heatmap_model, tr, va, crit_h, opt_h, device, epochs=30, save_path="results/heatmap_model.pth")

    # 2) Regression 模型
    tr2, va2 = get_loaders(mode="regression", batch_size=32)
    reg_model = RegressionNet(num_keypoints=5).to(device)
    crit_r = torch.nn.MSELoss()
    opt_r = torch.optim.Adam(reg_model.parameters(), lr=1e-3)
    train_loop(reg_model, tr2, va2, crit_r, opt_r, device, epochs=30, save_path="results/regression_model.pth")

if __name__ == "__main__":
    main()