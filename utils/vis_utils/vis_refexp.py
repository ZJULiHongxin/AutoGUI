import os, json, random, cv2, numpy as np
from datasets import load_dataset
def vis():
    compared = [{"model": "seeclick", "eval_file": "/mnt/nvme0n1p1/hongxin_li/AutoGUI/logs/seeclick/0925_101217_test_seeclick_func_pred_rec-motif-refexp-screenspot_rec-vwb/refexp_rec_test.json", "scale": 1},
             {"model": "autogui", "eval_file": "/mnt/nvme0n1p1/hongxin_li/AutoGUI/logs/autogui/0925_Qwen_AutoGUI702k_func_pred_rec-motif-refexp-screenspot_rec-vwb/refexp_rec_test.json", "scale": 100}]
    
    refexp_test = load_dataset("ivelin/ui_refexp_saved", split="test")
    
    for x in compared:
        x['eval_result'] = json.load(open(x['eval_file'], 'r'))["logs"]
    
    for i in range(len(refexp_test)):
        img = cv2.cvtColor(np.array(refexp_test[i]['image']), cv2.COLOR_RGB2BGR)
        H, W = img.shape[:2]
        norm_box = eval(refexp_test[i]['target_bounding_box'])
        unnorm_box = list(map(int, [norm_box['xmin'] * W, norm_box['ymin'] * H, norm_box['xmax'] * W, norm_box['ymax'] * H]))
        instruc = refexp_test[i]['prompt']

        cv2.rectangle(img, (unnorm_box[0], unnorm_box[1]), (unnorm_box[2], unnorm_box[3]), (0, 255, 0), 2)
        print()
        print(instruc)
        
        skip = False
        for model_info in compared:
            target_pred = eval(model_info["eval_result"][i]["resps"][0]["response"])

            target_pred = round(target_pred[0] / model_info['scale'] * W), round(target_pred[1] / model_info['scale'] * H)
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
    
    