import os, json, random, cv2, numpy as np
from datasets import load_dataset
def vis():
    compared = [{"model": "seeclick", "eval_file": "logs/seeclick/0925_101217_test_seeclick_func_pred_rec-motif-refexp-screenspot_rec-vwb/motif_point_test.json", "scale": 1},
             {"model": "autogui", "eval_file": "logs/autogui/0925_Qwen_AutoGUI702k_func_pred_rec-motif-refexp-screenspot_rec-vwb/motif_point_test.json", "scale": 100}]
    
    motif_test = load_dataset("HongxinLi/MOTIF-EVAL", split="test_au_tu")
    
    for x in compared:
        x['eval_result'] = json.load(open(x['eval_file'], 'r'))["logs"]
    
    for i in range(len(motif_test)):
        img = cv2.cvtColor(np.array(motif_test[i]['image']), code=cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        unnorm_box = motif_test[i]['ui_pos_obj_view_bbox']
        if unnorm_box is None: continue
        elem_text = motif_test[i]['goal']

        cv2.rectangle(img, (unnorm_box[0], unnorm_box[1]), (unnorm_box[2], unnorm_box[3]), (0, 255, 0), 2)
        print()
        print(elem_text)
        
        skip = False
        for model_info in compared:
            target_pred = list(map(float, model_info["eval_result"][i]["resps"][0]["response"].strip('()').split(',')))
            
            target_pred = round(target_pred[0] / model_info["scale"] * W), round(target_pred[1] / model_info["scale"] * H)
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
    
    