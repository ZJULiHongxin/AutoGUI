"""Grounding-task generation ported from ``generate_ft_data.py``.

Builds "locate the element supporting this functionality" grounding samples
from recorded trajectories, normalizes coordinates to the ``[0, 999]`` range,
and optionally publishes the resulting dataset to the Hugging Face Hub.
"""

import glob
import json
import os

import cv2
import datasets
from datasets import Dataset
from PIL import Image
from tqdm import tqdm


def normalize_coord(value: int, full: int) -> str:
    """Normalize ``value`` against ``full`` into a zero-padded ``[0, 999]`` string."""
    return f"{int(value / full * 1000):03d}"


def build_grounding_sample(
    *,
    func: str,
    rect: dict,
    scale: int,
    full_w: int,
    full_h: int,
    image_rel_path: str,
    step_id: int,
) -> dict:
    """Build a single grounding sample for one UI element.

    ``rect`` carries ``top``/``left``/``width``/``height``; coordinates are
    normalized to the ``[0, 999]`` range. Produces the ``{id, image, bbox,
    center, downsize, conversations}`` dict with a point-style GPT answer.
    """
    task = f"Please locate the element supporting this functionality: {func.rstrip('.')}."

    y1, x1, w, h = rect["top"], rect["left"], rect["width"], rect["height"]
    x2, y2 = x1 + w, y1 + h
    center_x, center_y = x1 + w / 2, y1 + h / 2

    norm_x1 = normalize_coord(x1, full_w)
    norm_y1 = normalize_coord(y1, full_h)
    norm_x2 = normalize_coord(x2, full_w)
    norm_y2 = normalize_coord(y2, full_h)
    target_bbox = f"[{norm_x1},{norm_y1},{norm_x2},{norm_y2}]"

    norm_center_x = normalize_coord(center_x, full_w)
    norm_center_y = normalize_coord(center_y, full_h)
    target_point = f"[{norm_center_x},{norm_center_y}]"

    return {
        "id": step_id,
        "image": image_rel_path,
        "bbox": target_bbox,
        "center": target_point,
        "downsize": scale,
        "conversations": [
            {"from": "human", "value": task},
            {"from": "gpt", "value": f"<point>{target_point}</point>"},
        ],
    }


def generate_tasks(
    *,
    gts_path,
    traj_folder,
    image_out_dir,
    meta_out_dir,
    full_w: int = 2560,
    full_h: int = 1440,
    scales=(1, 2, 4),
    push_to_hub: str | None = None,
) -> list[dict]:
    """Generate grounding tasks for every trajectory step across ``scales``.

    Reads ground-truth functionalities from ``gts_path``, resizes each
    pre-action screenshot into a per-scale subfolder under ``image_out_dir``,
    writes a per-scale JSON of samples under ``meta_out_dir``, and returns the
    accumulated meta records. When ``push_to_hub`` is a non-None repo id the
    assembled dataset is published there; otherwise nothing is uploaded.
    """
    with open(gts_path, "r") as f:
        gts = json.load(f)

    meta = []
    for scale in scales:
        samples = []

        scale_folder_name = f"ui_llava/downsize{scale}"
        image_folder_this_scale = os.path.join(image_out_dir, scale_folder_name)
        os.makedirs(image_folder_this_scale, exist_ok=True)

        for traj_name, gt in tqdm(gts.items(), total=len(gts), desc=f"Scale {scale}"):
            for step_id, func in gt.items():
                step_id = int(step_id)

                prev_step_image_file = [
                    x
                    for x in glob.glob(
                        os.path.join(traj_folder, traj_name, f"step{step_id-1:03d}*.png")
                    )
                    if "marked" not in x
                ][0]
                prev_state_meta_file = prev_step_image_file.replace(".png", "_meta.json")

                new_image_file_name = os.path.join(
                    image_folder_this_scale,
                    f"{traj_name}_{step_id-1}_downsize{scale}.png",
                )

                current_state_meta_file = glob.glob(
                    os.path.join(traj_folder, traj_name, f"step{step_id:03d}*_meta.json")
                )[0]

                action_marker = current_state_meta_file[
                    current_state_meta_file.find("_action")
                    + 7 : current_state_meta_file.find("_domain")
                ]

                with open(prev_state_meta_file, "r") as f:
                    nodes_info = json.load(f)
                    for node in nodes_info:
                        if node["hint_marker_text"] == action_marker:
                            rect = node["rect"]

                            image = cv2.imread(prev_step_image_file)
                            image = cv2.resize(
                                image,
                                (full_w // (2 * scale), full_h // (2 * scale)),
                                interpolation=cv2.INTER_AREA,
                            )
                            cv2.imwrite(new_image_file_name, image)

                            sample = build_grounding_sample(
                                func=func,
                                rect=rect,
                                scale=scale,
                                full_w=full_w,
                                full_h=full_h,
                                image_rel_path=f"{scale_folder_name}/{os.path.basename(new_image_file_name)}",
                                step_id=step_id,
                            )
                            break
                    else:
                        raise Exception("Invalid!")

                samples.append(sample)

                meta.append(
                    {
                        "file_name": os.path.basename(new_image_file_name),
                        "bbox": sample["bbox"],
                        "center": sample["center"],
                        "instruction": sample["conversations"][0]["value"],
                        "data_type": f"downsize{scale}",
                        "data_source": f"downsize{scale}",
                        "image": Image.fromarray(image),
                    }
                )

        os.makedirs(meta_out_dir, exist_ok=True)
        with open(os.path.join(meta_out_dir, f"ui_llava_downssize{scale}.json"), "w") as f:
            json.dump(samples, f, indent=2)

    if push_to_hub is not None:
        dataset = Dataset.from_generator(
            lambda: (
                {
                    "image": sample["image"],
                    "file_name": sample["file_name"],
                    "bbox": sample["bbox"],
                    "center": sample["center"],
                    "instruction": sample["instruction"],
                    "data_type": sample["data_type"],
                    "data_source": sample["data_source"],
                }
                for sample in meta
            ),
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "file_name": datasets.Value("string"),
                    "bbox": datasets.Value("string"),
                    "center": datasets.Value("string"),
                    "instruction": datasets.Value("string"),
                    "data_type": datasets.Value("string"),
                    "data_source": datasets.Value("string"),
                }
            ),
        )
        dataset.push_to_hub(push_to_hub, private=True)

    return meta
