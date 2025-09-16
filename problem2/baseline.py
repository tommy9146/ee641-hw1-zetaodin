# 可选：简单 ablation 占位（保存说明性文本，避免空文件）
import json, os
os.makedirs("results", exist_ok=True)
json.dump({"note":"Add ablations if time permits: heatmap size, sigma, skip connections"},
          open("results/ablation_todo.json","w"), indent=2)
print("Baseline placeholder saved to results/ablation_todo.json")