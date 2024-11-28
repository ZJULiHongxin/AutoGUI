import os, json, random, cv2, numpy as np
from datasets import load_dataset
def vis():
    compared = [{"model": "seeclick", "eval_file": "logs/seeclick/0925_101217_test_seeclick_func_pred_rec-motif-refexp-screenspot_rec-vwb/screenspot_rec_test.json"},
             {"model": "autogui", "eval_file": "logs/autogui/0925_Qwen_AutoGUI702k_func_pred_rec-motif-refexp-screenspot_rec-vwb/screenspot_rec_test.json"}]
    
    sspot_test = load_dataset("rootsautomation/ScreenSpot", split="test")
    
    for x in compared:
        x['eval_result'] = json.load(open(x['eval_file'], 'r'))["logs"]
    
    for i in range(len(sspot_test)):
        img = cv2.cvtColor(np.array(sspot_test[i]['image']), code=cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        norm_box = sspot_test[i]['bbox']
        unnorm_box = [round(norm_box[0] * W), round(norm_box[1] * H), round(norm_box[2] * W), round(norm_box[3] * H)]
        elem_text, elem_role = sspot_test[i]['instruction'], sspot_test[i]['data_type']

        cv2.rectangle(img, (unnorm_box[0], unnorm_box[1]), (unnorm_box[2], unnorm_box[3]), (0, 255, 0), 2)
        print()
        print(elem_text, elem_role)
        
        skip = False
        for model_info in compared:
            target_pred = model_info["eval_result"][i]["screenspot_Center_ACC"]["pred"]
            
            target_pred = round(target_pred[0] * W), round(target_pred[1] * H)
            is_corr = unnorm_box[0] <= target_pred[0] <= unnorm_box[2] and unnorm_box[1] <= target_pred[1] <= unnorm_box[3]
            
            if model_info['model'] == 'seeclick' and not is_corr:
                skip = True
                break
            
            print(model_info['model'], target_pred, is_corr, end=' | ')
            
            cv2.circle(img, target_pred, 6, (0, 0, 255), 2)
            cv2.circle(img, target_pred, 3, (0, 0, 255), -1)
        
        if skip: continue
        cv2.imwrite("test.png", img)
        1+1
            
    
vis()
    
    