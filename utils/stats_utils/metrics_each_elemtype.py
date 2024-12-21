# 用于计算bmk上的测评结果在各个元素类别上的分值
import json, os
from tqdm import tqdm
from pprint import pprint
import numpy as np

def calc_funcpred():
    eval_file = "/data0/jingran/workspace/hongxin_li/AutoGUI/logs/slime/0924_191703_test_slime_func_pred_rec-motif-refexp-screenspot_rec-vwb/func_pred_rec_test.json"
    
    eval_results = json.load(open(eval_file))['logs']
    
    platform_acc = {'web': {'acc':[], 'type': [], 'diff': [], 'area': []}, 'mobile': {'acc':[], 'type': [], 'diff': [], 'area': []}}

    for item in tqdm(eval_results, total=len(eval_results)):
        elem_role = item["doc"]["elem_role"]
        platform = item["Center_ACC"]["device"]
        
        platform_acc[platform]['type'].append(elem_role)

        pred_target, gt_bbox = item["Center_ACC"]["pred"], item["Center_ACC"]["bbox"]
        
        if gt_bbox[0] <= pred_target[0] <= gt_bbox[2] and gt_bbox[1] <= pred_target[1] <= gt_bbox[3]:
            platform_acc[platform]['acc'].append(True)
        else: platform_acc[platform]['acc'].append(False)

        gt_center = np.array([(gt_bbox[0]+gt_bbox[2])/2, (gt_bbox[1]+gt_bbox[3])/2])
        diff = np.linalg.norm(gt_center - np.array(pred_target)).item()
        
        platform_acc[platform]['diff'].append(diff)
        platform_acc[platform]['area'].append((gt_bbox[2]-gt_bbox[0]) * (gt_bbox[3]-gt_bbox[1]))

    for platform, info in platform_acc.items():
        print(platform)
        elem_type_acc = {}
        for is_corr, type in zip(info['acc'], info['type']):
            if type not in elem_type_acc: elem_type_acc[type] = [0,0]
            elem_type_acc[type][1] += 1
            
            if is_corr: elem_type_acc[type][0] += 1

        for type, stats in elem_type_acc.items(): 
            a,b = stats
            print(f"{type}: {a} / {b} = {a/b:.3f}")

        total_acc, total_samples = sum(info['acc']), len(info['acc'])
        print(f"Overall: {total_acc} / {total_samples} = {total_acc / total_samples if total_samples else 0:.3f}")
        print(np.median(platform_acc[platform]['diff']))
    
    # 计算元素面积与准确度的关系
    for platform, info in platform_acc.items():
        ranges = np.concatenate([np.linspace(0, 0.01, 6), [1e6]])

        bins = [[] for _ in range(len(ranges))]
        for is_corr, area in zip(info['acc'], info['area']):
            bin_i = 0
            while bin_i < len(ranges):
                if ranges[bin_i] <= area < ranges[bin_i+1]:
                    break
                bin_i += 1

            bins[bin_i].append(is_corr)
        
        print("{}\nBins: {}\nAcc.: {}".format(platform, '\t'.join(f'{x:.3f} ({len(bins[bin_i])})' for bin_i, x in enumerate(ranges)), '\t'.join('{:.3f}'.format(sum(b)/len(b) if len(b) > 0 else 0.0)  for b in bins)))
        
    # pprint(elem_type_acc)
    # pprint(platform_acc)

calc_funcpred()