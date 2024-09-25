import numpy as np
import lxml, re, copy
from lxml import etree
from difflib import SequenceMatcher

def compute_target_overlap_ratio(box1, box2):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    - box1 (list of float): Bounding box [x_min, y_min, x_max, y_max].
    - box2 (list of float): Bounding box [x_min, y_min, x_max, y_max].

    Returns:
    - float: IoU of box1 and box2.
    """
    # Determine the coordinates of the intersection rectangle
    
    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    # If the box has no area
    if box1[0] == box1[2] or box1[1] == box1[3]:
        if box2[0] < (box1[0] + box1[2]) / 2 < box2[2] and box2[1] < (box1[1] + box1[3]) / 2 < box2[3]:
            return 1.0
        else:
            return 0.0

    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Compute the area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])

    return intersection_area / box1_area

from PIL import Image, ImageDraw

def draw_dashed_rectangle(draw, coordinates, outline_color, dash_length=5):
    """
    Draws a dashed rectangle on the given ImageDraw object.
    
    Parameters:
    draw: ImageDraw object.
    coordinates: List of four integers [left, top, right, bottom].
    outline_color: Color of the outline.
    dash_length: Length of the dashes.
    """
    left, top, right, bottom = coordinates
    
    # Top border
    for x in range(left, right, dash_length * 2):
        draw.line([(x, top), (min(x + dash_length, right), top)], fill=outline_color)
    
    # Bottom border
    for x in range(left, right, dash_length * 2):
        draw.line([(x, bottom), (min(x + dash_length, right), bottom)], fill=outline_color)
    
    # Left border
    for y in range(top, bottom, dash_length * 2):
        draw.line([(left, y), (left, min(y + dash_length, bottom))], fill=outline_color)
    
    # Right border
    for y in range(top, bottom, dash_length * 2):
        draw.line([(right, y), (right, min(y + dash_length, bottom))], fill=outline_color)

def get_target_center_screenshot(screenshot, max_h, max_w, gt_box):
    W, H = screenshot.size

    raw_upper = (gt_box[1] + gt_box[3]) / 2 - max_h // 2
    raw_lower = (gt_box[1] + gt_box[3]) / 2 + max_h // 2
    if raw_upper < 0:
        upper, lower = 0, max_h
    elif raw_lower > H:
        upper, lower = H - max_h, H
    else:
        upper = round(raw_upper)
        lower = upper + max_h

    raw_left = (gt_box[0] + gt_box[2]) / 2 - max_w // 2
    raw_right = (gt_box[0] + gt_box[2]) / 2 + max_w // 2
    if raw_left < 0:
        left, right = 0, max_w
    elif raw_right > W:
        left, right = W - max_w, W
    else:
        left = round(raw_left)
        right = left + max_w
    
    cropped_image = screenshot.crop([left, upper, right, lower])

    return cropped_image, left, upper, right, lower

def parse_action(text):
    if text.startswith('(') and text.endswith(')'):
        action_type, target = 'click', list(map(float, text[1:-1].split(',')))
    else:
        action_type_start = text.find(' ', text.find("action_type")) + 1
        action_type_end = text.find(',', action_type_start)
        
        action_type = text[action_type_start:action_type_end]
        
        if action_type == 'click':
            target_start = text.find(' ', text.find('target', action_type_end)) + 2
            target_end = text.find(')', target_start)
            target = list(map(float, text[target_start:target_end].split(',')))
        else:
            target = None
    
    return action_type, target

def test_inside_box(pred_point, gt_point, box_sizes):
    return [abs(pred_point[0] - gt_point[0]) < size and abs(pred_point[1] - gt_point[1]) < size for size in box_sizes]

def get_attribute_repr(node, max_value_length=5, max_length=20):
    # get attribute values in order
    attr_values_set = set()
    attr_values = ""
    for attr in [
        "role",
        "aria_role",
        "type",
        "alt",
        "aria_description",
        "aria_label",
        "label",
        "title",
        "name",
        "text_value",
        "value",
        "placeholder",
        "input_checked",
        "input_value",
        "option_selected",
        "class",
    ]:
        if attr in node.attrib and node.attrib[attr] is not None:
            value = node.attrib[attr].lower()
            # less menaingful values
            if value in [
                "hidden",
                "none",
                "presentation",
                "null",
                "undefined",
            ] or value.startswith("http"):
                continue
            value = value.split()
            value = " ".join([v for v in value if len(v) < 15][:max_value_length])
            if value and value not in attr_values_set:
                attr_values_set.add(value)
                attr_values += value + " "
    uid = node.attrib.get("backend_node_id", "")
    # clear all attributes
    node.attrib.clear()
    if uid:
        node.attrib["id"] = uid
    # add meta attribute
    if attr_values:
        node.attrib["meta"] = " ".join(attr_values.split()[:max_length])


def get_descendants(node, max_depth, current_depth=0):
    if current_depth > max_depth:
        return []

    descendants = []
    for child in node:
        descendants.append(child)
        descendants.extend(get_descendants(child, max_depth, current_depth + 1))

    return descendants

def postprocess_action_llm(text):
    # C.
    # Action: SELECT
    # Value: Queen
    text = text.strip()
    selected_option = re.search(r"Answer: ([A-Z])", text)
    selected_option = (
        selected_option.group(1) if selected_option is not None else "A"
    )
    action = re.search(r"Action: (CLICK|SELECT|TYPE)", text)
    action = action.group(1) if action is not None else ""
    value = re.search(r"Value: (.*)$", text, re.MULTILINE)
    value = value.group(1) if value is not None else ""
    return selected_option, action.strip() + " " + value.strip()
    
def prune_tree(
    dom_tree,
    candidate_set,
    max_depth=5,
    max_children=50,
    max_sibling=3,
):
    nodes_to_keep = set()
    for candidate_id in candidate_set:
        candidate_node = dom_tree.xpath(f'//*[@backend_node_id="{candidate_id}"]')[0]
        nodes_to_keep.add(candidate_node.attrib["backend_node_id"])
        # get all ancestors
        nodes_to_keep.update(
            [
                x.attrib.get("backend_node_id", "")
                for x in candidate_node.xpath("ancestor::*")
            ]
        )
        # get descendants with max depth
        nodes_to_keep.update(
            [
                x.attrib.get("backend_node_id", "")
                for x in get_descendants(candidate_node, max_depth)
            ][:max_children]
        )
        # get siblings within range
        parent = candidate_node.getparent()
        if parent is not None:
            siblings = [x for x in parent.getchildren() if x.tag != "text"]
            idx_in_sibling = siblings.index(candidate_node)
            nodes_to_keep.update(
                [
                    x.attrib.get("backend_node_id", "")
                    for x in siblings[
                        max(0, idx_in_sibling - max_sibling) : idx_in_sibling
                        + max_sibling
                        + 1
                    ]
                ]
            )
    # clone the tree
    new_tree = copy.deepcopy(dom_tree)
    # remove nodes not in nodes_to_keep
    for node in new_tree.xpath("//*")[::-1]:
        if node.tag != "text":
            is_keep = node.attrib.get("backend_node_id", "") in nodes_to_keep
            is_candidate = node.attrib.get("backend_node_id", "") in candidate_set
        else:
            is_keep = (
                node.getparent().attrib.get("backend_node_id", "") in nodes_to_keep
            )
            is_candidate = (
                node.getparent().attrib.get("backend_node_id", "") in candidate_set
            )
        if not is_keep and node.getparent() is not None:
            node.getparent().remove(node)
        else:
            if not is_candidate or node.tag == "text":
                node.attrib.pop("backend_node_id", None)
            if (
                len(node.attrib) == 0
                and not any([x.tag == "text" for x in node.getchildren()])
                and node.getparent() is not None
                and node.tag != "text"
                and len(node.getchildren()) <= 1
            ):
                # insert all children into parent
                for child in node.getchildren():
                    node.addprevious(child)
                node.getparent().remove(node)
    return new_tree

def get_tree_repr(
    tree, max_value_length=5, max_length=20, id_mapping={}, keep_html_brackets=False
):
    if isinstance(tree, str):
        tree = etree.fromstring(tree)
    else:
        tree = copy.deepcopy(tree)
    for node in tree.xpath("//*"):
        if node.tag != "text":
            if "backend_node_id" in node.attrib:
                if node.attrib["backend_node_id"] not in id_mapping:
                    id_mapping[node.attrib["backend_node_id"]] = len(id_mapping)
                node.attrib["backend_node_id"] = str(
                    id_mapping[node.attrib["backend_node_id"]]
                )
            get_attribute_repr(node, max_value_length, max_length)
        else:
            node.text = " ".join(node.text.split()[:max_length])
    tree_repr = etree.tostring(tree, encoding="unicode")

    tree_repr = tree_repr.replace('"', " ")
    tree_repr = (
        tree_repr.replace("meta= ", "").replace("id= ", "id=").replace(" >", ">")
    )
    tree_repr = re.sub(r"<text>(.*?)</text>", r"\1", tree_repr)
    if not keep_html_brackets:
        tree_repr = tree_repr.replace("/>", "$/$>")
        tree_repr = re.sub(r"</(.+?)>", r")", tree_repr)
        tree_repr = re.sub(r"<(.+?)>", r"(\1", tree_repr)
        tree_repr = tree_repr.replace("$/$", ")")

    html_escape_table = [
        ("&quot;", '"'),
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
        ("&ndash;", "-"),
        ("&rsquo;", "'"),
        ("&lsquo;", "'"),
        ("&ldquo;", '"'),
        ("&rdquo;", '"'),
        ("&#39;", "'"),
        ("&#40;", "("),
        ("&#41;", ")"),
    ]
    for k, v in html_escape_table:
        tree_repr = tree_repr.replace(k, v)
    tree_repr = re.sub(r"\s+", " ", tree_repr).strip()

    return tree_repr, id_mapping

def format_input_multichoice(
    sample, candidate_ids, gt=-1, previous_k=5, keep_html_brackets=False
):
    # 之前dom_tree作为输入参数，这可能是导致性能减半的原因
    dom_tree = lxml.etree.fromstring(sample["cleaned_html"])
    dom_tree = prune_tree(dom_tree, candidate_ids)
    tree_repr, id_mapping = get_tree_repr(
        dom_tree, id_mapping={}, keep_html_brackets=keep_html_brackets
    )
    candidate_nodes = dom_tree.xpath("//*[@backend_node_id]")
    choices = []
    for idx, node in enumerate(candidate_nodes):
        choice_repr = " ".join(
                    get_tree_repr(
                        node,
                        id_mapping=id_mapping,
                        keep_html_brackets=keep_html_brackets,
                    )[0].split()[:10]
                )

        next_new_elem_id = choice_repr.find('<', choice_repr.find('</'))
        if choice_repr[next_new_elem_id+1] != '/': continue

        choices.append(
            [
                node.attrib["backend_node_id"],
                choice_repr,
            ]
        )
    gt = id_mapping.get(gt, -1)
    seq_input = (
        "Based on the HTML webpage above, try to complete the following task:\n"
        f"Task: {sample['confirmed_task']}\n"
        f"Previous actions:\n"
    )
    if len(sample["previous_actions"]) > 0:
        for action in sample["previous_actions"][-previous_k:]:
            seq_input += f"{action}\n"
    else:
        seq_input += "None\n"
    seq_input += (
        "What should be the next action? Please select from the following choices "
        "(If the correct action is not in the page above, please select A. 'None of the options below'):\n\n"
        "A. None of the options below\n"
    )

    if gt == -1:
        seq_target = "A."
    else:
        gt += 1
        current_action_op = sample["operation"]["op"]
        current_action_value = sample["operation"]["value"]
        seq_target = f"{chr(65 + gt)}.\n" f"Action: {current_action_op}\n"
        if current_action_op != "CLICK":
            seq_target += f"Value: {current_action_value}"
    return tree_repr, seq_input, seq_target, choices

def postprocess_action(text, choices=None):
    # C.
    # Action: SELECT
    # Value: Queen

    if choices is None:
        text = text.strip()
        selected_option = text[0]
        action = re.search(r"Action: (CLICK|SELECT|TYPE)", text)
        action = action.group(1) if action is not None else ""
        value = re.search(r"Value: (.*)$", text, re.MULTILINE)
        value = value.group(1) if value is not None else ""
        return selected_option, action.strip() + " " + value.strip()
    else:
        text = text.strip()
        if text.startswith("None"):
            selected_option = None
        else:
            selected_option = re.search(r"Element: (.*)$", text, re.MULTILINE)
            selected_option = (
                selected_option.group(1) if selected_option is not None else ""
            )
            selected_id = re.search(r"id=(\d+)", selected_option)
            if selected_id is not None:
                selected_id = selected_id.group(1)
                selected_id = int(selected_id)
                if selected_id >= len(choices):
                    selected_id = None
            if selected_id is None:
                # try matching by text
                choice_matching_scores = [
                    SequenceMatcher(None, selected_option, choice).ratio()
                    for choice in choices
                ]
                selected_id = np.argmax(choice_matching_scores)
            selected_option = choices[selected_id][0]

        action = re.search(r"Action: (CLICK|SELECT|TYPE)", text)
        action = action.group(1) if action is not None else ""
        value = re.search(r"Value: (.*)$", text, re.MULTILINE)
        value = value.group(1) if value is not None else ""
        return selected_option, action.strip() + " " + value.strip()


def calculate_f1(pred, label):
    pred = set(pred.strip().split())
    label = set(label.strip().split())
    if len(pred) == 0 and len(label) == 0:
        return 1
    if len(pred) == 0 or len(label) == 0:
        return 0

    tp = len(pred & label)
    fp = len(pred - label)
    fn = len(label - pred)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision == 0 or recall == 0:
        return 0
    f1 = 2 * precision * recall / (precision + recall)
    return f1