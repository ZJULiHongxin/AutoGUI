import datasets, io
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw

def check_motif():
    from lmms_eval.tasks.ui_motif.utils_point import motif_preprocess_dataset, motif_doc_to_visual, motif_doc_to_text, motif_doc_to_target
    
    ds = datasets.load_dataset("HongxinLi/MOTIF-EVAL", split='test_au_tu')
    
    proc_ds = motif_preprocess_dataset(ds)

    for sample in tqdm(proc_ds, total=len(proc_ds)):
        image = motif_doc_to_visual(sample)[0]
        prompt = motif_doc_to_text(sample)
        
        norm_box, action = motif_doc_to_target(sample)
        
        W, H = image.size
        
        draw = ImageDraw.Draw(image)
        
        x1, y1, x2, y2 = round(norm_box[0] * W), round(norm_box[1] * H), round(norm_box[2] * W), round(norm_box[3] * H)
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        image.save("test.png")
        print(f"Prompt: {prompt}\nAction: {action}")
        1+1

check_motif()