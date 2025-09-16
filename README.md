# EE641 Homework 1: Multi-Scale Detection and Spatial Regression

Author: Zetao Ding  
Email: zetaodin@usc.edu  

---

##  Environment Setup

Recommended environment:
- Python 3.9+
- PyTorch 2.0+ (tested with 2.8.0 on Colab, CUDA 12.4)
- numpy
- matplotlib
- pillow

Setup with `venv`:
```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
# .venv\Scripts\activate    # Windows PowerShell
pip install torch torchvision numpy matplotlib pillow
```

---

##  Dataset Generation

Use the provided script to generate both **detection** and **keypoint** datasets:
```bash
python generate_datasets.py --seed 641 --num_train 1000 --num_val 200
```

Datasets will be saved in:
```
datasets/
├── detection/
│   ├── train/ (1000 images)
│   ├── val/   (200 images)
│   ├── train_annotations.json
│   └── val_annotations.json
└── keypoints/
    ├── train/ (1000 images)
    ├── val/   (200 images)
    ├── train_annotations.json
    └── val_annotations.json
```

---

##  Problem 1: Multi-Scale Detection

### Training
```bash
cd problem1
python train.py
```

### Evaluation
```bash
python evaluate.py
```

### Outputs
- Model: `problem1/results/best_model.pth`
- Logs: `problem1/results/training_log.json`
- Visualizations: `problem1/results/visualizations/*.png`
- Evaluation: mAP@0.5 reported in console

---

##  Problem 2: Spatial Regression for Keypoints

Two approaches are implemented:
1. **Heatmap-based regression**
2. **Direct coordinate regression**

### Training
```bash
cd problem2
python train.py
```

### Evaluation
```bash
python evaluate.py
```

### Outputs
- Models:  
  - `problem2/results/heatmap_model.pth`  
  - `problem2/results/regression_model.pth`
- Logs: `problem2/results/training_log.json`
- Visualizations: `problem2/results/visualizations/`
- Evaluation: PCK@0.2 curve

---

##  Results to Include in Report
- **Problem 1**:  
  - Training/validation loss curve  
  - mAP values  
  - Visualization of predicted vs ground truth boxes  

- **Problem 2**:  
  - Loss curves (Heatmap vs Regression)  
  - PCK@0.2 curve  
  - Keypoint prediction visualizations  
  - Discussion of differences between two approaches  

---

##  Notes
- All experiments use **seed=641** for reproducibility.  
- Training time (Colab T4 GPU):  
  - Problem 1 ≈ 12–15 min (50 epochs)  
  - Problem 2 ≈ 5–7 min (30+30 epochs)  
- Report all results in `report.pdf`.
