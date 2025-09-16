import os, json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class KeypointDataset(Dataset):
    """
    读取 128x128 灰度图 + 5 个关键点 (x,y) 像素坐标
    output_type: 'heatmap' or 'regression'
    兼容多种注释结构：
      1) {"images":[...], "annotations":[{"image_id":..,"keypoints":[[x,y],...]}]}
      2) {"images":[...], "annotations":{"000001.png":[[x,y],...], ...}}
      3) {"images":[{"id":..,"file_name":"...","keypoints":[[x,y],...]}, ...]}
    """
    def __init__(self, image_dir, annotation_file, output_type='heatmap',
                 heatmap_size=64, sigma=2.0):
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        with open(annotation_file, 'r') as f:
            ann = json.load(f)

        # --- 统一出 images 列表：[{id, file_name, width, height}, ...]
        assert "images" in ann, "annotations json 缺少 'images' 键"
        self.images = ann["images"]
        # 如果 images 里没有 id，则补一个顺序 id
        for i, im in enumerate(self.images):
            if "id" not in im:
                im["id"] = i

        self.id2img = {im["id"]: im for im in self.images}
        self.img_ids = [im["id"] for im in self.images]

        # --- 统一出每张图的关键点 ndarray [5,2]
        kpts_by_file = {}

        if "annotations" in ann:
            if isinstance(ann["annotations"], list):
                # 1) COCO-ish 列表，按 image_id 匹配
                by_imgid = {}
                for a in ann["annotations"]:
                    # 容忍不同字段名
                    kp = a.get("keypoints") or a.get("kpts") or a.get("points")
                    if kp is None:
                        continue
                    by_imgid[a["image_id"]] = np.array(kp, dtype=np.float32)
                for im in self.images:
                    arr = by_imgid.get(im["id"], np.zeros((5,2), dtype=np.float32))
                    kpts_by_file[im["file_name"]] = arr
            elif isinstance(ann["annotations"], dict):
                # 2) 映射：file_name -> [[x,y],...]
                for im in self.images:
                    kp = ann["annotations"].get(im["file_name"])
                    if kp is None:
                        kpts_by_file[im["file_name"]] = np.zeros((5,2), dtype=np.float32)
                    else:
                        kpts_by_file[im["file_name"]] = np.array(kp, dtype=np.float32)
            else:
                raise ValueError("未知的 annotations 类型")
        else:
            # 3) 关键点直接在 images 里
            for im in self.images:
                kp = im.get("keypoints") or im.get("kpts") or im.get("points")
                if kp is None:
                    kpts_by_file[im["file_name"]] = np.zeros((5,2), dtype=np.float32)
                else:
                    kpts_by_file[im["file_name"]] = np.array(kp, dtype=np.float32)

        # 存成 id->kpts
        self.kpts_by_img = {}
        for im in self.images:
            self.kpts_by_img[im["id"]] = kpts_by_file[im["file_name"]].astype(np.float32)

    def __len__(self): 
        return len(self.img_ids)

    def _gen_heatmap(self, keypoints, H, W):
        """生成 [5,H,W] 高斯热力图"""
        K = keypoints.shape[0]
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        heatmaps = []
        sx = W / 128.0
        sy = H / 128.0
        kx = keypoints[:, 0] * sx
        ky = keypoints[:, 1] * sy
        for i in range(K):
            g = np.exp(-((xx - kx[i])**2 + (yy - ky[i])**2) / (2*(self.sigma**2)))
            heatmaps.append(g.astype(np.float32))
        return torch.from_numpy(np.stack(heatmaps, 0))

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        meta = self.id2img[img_id]
        path = os.path.join(self.image_dir, meta["file_name"])
        img = Image.open(path).convert("L")  # [H,W]
        arr = np.array(img, dtype=np.float32)/255.0
        img_t = torch.from_numpy(arr).unsqueeze(0)   # [1,128,128]

        kpts = self.kpts_by_img[img_id].copy()       # [5,2] in pixel
        if self.output_type == 'heatmap':
            targets = self._gen_heatmap(kpts, self.heatmap_size, self.heatmap_size)  # [5,64,64]
        else:
            # 归一化到 [0,1]
            targets = torch.from_numpy(
                np.stack([kpts[:,0]/128.0, kpts[:,1]/128.0], -1).reshape(-1).astype(np.float32)
            )
        return img_t, targets