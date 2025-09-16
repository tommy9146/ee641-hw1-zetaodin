import json, os
from PIL import Image
import torch
from torch.utils.data import Dataset

class ShapeDetectionDataset(Dataset):
    """
    读取 COCO 风格标注：
    images: [{id, file_name, width, height}, ...]
    annotations: [{image_id, category_id, bbox[x,y,w,h]}, ...]
    """
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            coco = json.load(f)
        self.id2img = {im["id"]: im for im in coco["images"]}
        self.img_ids = [im["id"] for im in coco["images"]]
        self.img_ann = {img_id: [] for img_id in self.img_ids}
        for ann in coco["annotations"]:
            self.img_ann[ann["image_id"]].append(ann)
        # 类别从 0 开始（0:circle,1:square,2:triangle）
        self.num_classes = 3

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        meta = self.id2img[img_id]
        path = os.path.join(self.image_dir, meta["file_name"])
        img = Image.open(path).convert("RGB")

        anns = self.img_ann[img_id]
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])  # [x1,y1,x2,y2]
            labels.append(int(a["category_id"]))  # 已是 0,1,2

        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0,4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

        if self.transform:
            img = self.transform(img)
        else:
            # to tensor [0,1]
            img = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                                    .view(img.size[1], img.size[0], 3).numpy().astype('float32')/255.0)).permute(2,0,1)

        targets = {"boxes": boxes, "labels": labels}
        return img, targets

def collate_fn(batch):
    imgs, tgs = zip(*batch)
    return torch.stack(imgs, 0), list(tgs)