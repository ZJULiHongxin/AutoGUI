import json, cv2, os, random

LOGDIR = '/mnt/nvme0n1p1/hongxin_li/AutoGUI/logs'
AITW_IMAGE_DIR = '/mnt/vdb1/hongxin.li/AITW/aitw_images'
logname = '/mnt/nvme0n1p1/hongxin_li/AutoGUI/logs/uipro/1103_153912_test_uipro_aitw/aitw_test.json'
SCALE = 100

eval_result = json.load(open(logname))['logs']

for sample in eval_result:
    img = cv2.imread(os.path.join(AITW_IMAGE_DIR, sample['doc']['image']))
    H,W=img.shape[:2]
    pred_action = sample['AITW_METRIC']['pred_acion']
    
    if pred_action['action_type'] == 'click':
        target_x, target_y = round(pred_action['attr']['target'][0]*W), round(pred_action['attr']['target'][1]*H)
    
        cv2.circle(img, center=(target_x, target_y), radius=5, color=(0,0,255), thickness=2)
        cv2.circle(img, center=(target_x, target_y), radius=2, color=(0,0,255), thickness=-1)
    
    print('Pred: ', sample['AITW_METRIC']['pred_acion'])
    print('GT: ', sample['AITW_METRIC']['gt_action'])
    cv2.imwrite('test.png', img)
    1+1