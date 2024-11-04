from PIL import Image
import re


def prepare_inputs(image_path, text, processor):
    img = Image.open(image_path).convert("RGB")
    size = img.size

    input_text = ("What to do to execute the command? " + text.strip()).lower()

    encoding = processor(
        images=img,
        text=input_text,
        return_tensors="pt",
        do_resize=True,
    )
    encoding["image_size"] = size

    return encoding


def postprocess(text: str, image_size: tuple[int]):
    """Function that decodes model's generation into action json.

    Args:
        text: single generated sample
        image_size: corresponding image size
    """
    pattern = r"</s><s>(<[^>]+>|[^<\s]+)\s*([^<]*?)(<loc_\d+>.*)"
    point_pattern = r"<loc_(\d+)><loc_(\d+)>"


    try:
        location = re.findall(point_pattern, text)[0]
        if len(location) > 0:
            point = [int(loc) for loc in location]

    except Exception:
        point = (0, 0)

    return point