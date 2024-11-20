import os, json, random, cv2, numpy as np
from datasets import load_dataset
def vis():
    compared = [{"model": "seeclick", "eval_file": "/mnt/nvme0n1p1/hongxin_li/AutoGUI/logs/seeclick/0925_101217_test_seeclick_func_pred_rec-motif-refexp-screenspot_rec-vwb/func_pred_rec_test.json"},
             {"model": "autogui", "eval_file": "/mnt/nvme0n1p1/hongxin_li/AutoGUI/logs/autogui/0925_Qwen_AutoGUI702k_func_pred_rec-motif-refexp-screenspot_rec-vwb/func_pred_rec_test.json"}]
    
    funcpred_test = load_dataset("WebAgent/AutoGUI-v1-test", split="test")
    
    for x in compared:
        x['eval_result'] = json.load(open(x['eval_file'], 'r'))["logs"]
    
    for i in range(len(funcpred_test)):
        img = np.array(funcpred_test[i]['image'])
        H, W = img.shape[:2]
        unnorm_box = list(map(int, funcpred_test[i]['unnormalized_box']))
        elem_text, elem_role, func = funcpred_test[i]['elem_text'], funcpred_test[i]['elem_role'], funcpred_test[i]["func"]

        cv2.rectangle(img, (unnorm_box[0], unnorm_box[1]), (unnorm_box[2], unnorm_box[3]), (0, 255, 0), 2)
        print()
        print(elem_text, elem_role, func)
        
        skip = False
        for model_info in compared:
            target_pred = model_info["eval_result"][i]["Center_ACC"]["pred"]
            
            target_pred = round(target_pred[0] * W), round(target_pred[1] * H)
            is_corr = unnorm_box[0] <= target_pred[0] <= unnorm_box[2] and unnorm_box[1] <= target_pred[1] <= unnorm_box[3]
            
            if model_info['model'] == 'autogui' and not is_corr:
                skip = True
                break
            
            print(model_info['model'], target_pred, is_corr, end=' | ')
            
            cv2.circle(img, target_pred, 6, (0, 0, 255), 2)
            cv2.circle(img, target_pred, 3, (0, 0, 255), -1)
        
        if skip: continue
        cv2.imwrite("test.png", img)
        1+1
            
    
vis()
    
    