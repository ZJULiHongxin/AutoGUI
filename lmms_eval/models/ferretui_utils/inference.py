import subprocess
import os
import subprocess
from PIL import Image, ImageDraw
import re
import json
import subprocess

def process_inference_results(results, process_image=False):
    """
    Process the inference results by:
    1. Adding bounding boxes on the image based on the coordinates in 'text'.
    2. Extracting and returning the text prompt.
    
    :param results: List of inference results with bounding boxes in 'text'.
    :return: (image, text)
    """
    processed_images = []
    extracted_texts = []

    for result in results:
        image_path = result['image_path']
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        bbox_str = re.search(r'\[\[([0-9,\s]+)\]\]', result['text'])
        if bbox_str:
            bbox = [int(coord) for coord in bbox_str.group(1).split(',')]
            x1, y1, x2, y2 = bbox

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        extracted_texts.append(result['text'])

        processed_images.append(img)

    if process_image:
        return processed_images, extracted_texts

    return extracted_texts

def inference_and_run(image_path, prompt, conv_mode="ferret_gemma_instruct", model_path="jadechoghari/Ferret-UI-Gemma2b", box=None, process_image=False):
    """
    Run the inference and capture the errors for debugging.
    """
    data_input = [{
        "id": 0,
        "image": os.path.basename(image_path),
        "image_h": Image.open(image_path).height,
        "image_w": Image.open(image_path).width,
        "conversations": [{"from": "human", "value": f"<image>\n{prompt}"}]
    }]
    
    if box:
        data_input[0]["box_x1y1x2y2"] = [[box]]

    with open("eval.json", "w") as json_file:
        json.dump(data_input, json_file)
    
    print("eval.json file created successfully.")
    
    cmd = [
        "python", "-m", "model_UI", 
        "--model_path", model_path,
        "--data_path", "eval.json", 
        "--image_path", ".", 
        "--answers_file", "eval_output.jsonl", 
        "--num_beam", "1", 
        "--max_new_tokens", "32",
        "--conv_mode", conv_mode
    ]

    if box:
        cmd.extend(["--region_format", "box", "--add_region_feature"])

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Subprocess output:\n{result.stdout}")
        print(f"Subprocess error (if any):\n{result.stderr}")
        print(f"Inference completed. Output written to eval_output.jsonl")

        output_folder = 'eval_output.jsonl'
        if os.path.exists(output_folder):
            json_files = [f for f in os.listdir(output_folder) if f.endswith(".jsonl")]
            if json_files:
                output_file_path = os.path.join(output_folder, json_files[0])
                with open(output_file_path, "r") as output_file:
                    results = [json.loads(line) for line in output_file]
                
                return process_inference_results(results, process_image)
            else:
                print("No output JSONL files found.")
                return None, None
        else:
            print("Output folder not found.")
            return None, None

    except subprocess.CalledProcessError as e:
        print(f"Error occurred during inference:\n{e}")
        print(f"Subprocess output:\n{e.output}")
        return None, None
