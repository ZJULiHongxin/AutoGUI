import matplotlib.pyplot as plt
import os, json
import numpy as np
from datasets import load_dataset
# 20.8
data = load_dataset("HongxinLi/VWB-EVAL", split="test")


# data = json.load(open("/mnt/nvme0n1p1/hongxin_li/AutoGUI/logs/uipro/1101_205231_test_uipro_func_pred_rec-motif-refexp-screenspot_rec-vwb/screenspot_rec_test.json")) # 16382

diffs = []

for x in data["logs"]:
    pred_target = np.array(x["screenspot_Center_ACC"]["pred"])[:2]
    gt_target = x["target"]
    gt_target = np.array([(gt_target[0]+gt_target[2])/2, (gt_target[1]+gt_target[3])/2])
    
    diff = np.linalg.norm(pred_target-gt_target)
    diffs.append(diff)
    

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(diffs, bins=30, color='blue', alpha=0.7)
plt.title('Distribution of UI-Pro pretrained with 20.8M', fontsize=14)
plt.xlim([0,1.0])
plt.xlabel('Normalized Distance', fontsize=14)
plt.ylabel('Count', fontsize=14)

# Enlarge tick labels
plt.tick_params(axis='both', which='major', labelsize=12)
# Show the plot
plt.savefig("20p8M.png")
