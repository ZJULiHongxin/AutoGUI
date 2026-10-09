"""Android XML -> text-tree parsing.

Ported verbatim from the XML half of
``WebpageFunctionality/utils/process_axtree.py`` (XML functions + helpers) and
``find_all_elem_texts_boxes`` from ``WebpageFunctionality/utils/tools.py``.

Only the XML-processing functions and their direct helpers are included here;
no LLM/network/image code is pulled in.
"""

import json, os, re
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element


# ---------------------------------------------------------------------------
# Language detection helpers (used by process_xml when skip_lang=True).
# Ported verbatim from tools.py.
# ---------------------------------------------------------------------------
def contains_japanese(text):
    return bool(re.search(r'[\u3040-\u30FF\uFF00-\uFFEF\u30A0-\u30FF]', text))


def contains_russian(text):
    return bool(re.search(r'[\u0400-\u04FF]', text))


def contains_arabic(text):
    return bool(re.search(r'[\u0600-\u06FF]', text))


def detect_invalid_lang(text):
    return contains_japanese(text) or contains_russian(text) or contains_arabic(text)


# ---------------------------------------------------------------------------
# XML parsing (ported verbatim from process_axtree.py).
# ---------------------------------------------------------------------------
DEFAULT_NODE_TAG = 'android.widget.TextView'
CORRECT_NESTED_TAG_CLASS_PATTERN = r'(<[^/>]*?)\$([^>]*?>)|(<\/[^>]*?)\$([^>]*?>)'


def preprocess_xml_content(xml_content):
    if '< index=' in xml_content:
        xml_content = xml_content.replace('< index=', f'<{DEFAULT_NODE_TAG} index=').replace('</>', f'</{DEFAULT_NODE_TAG}>')

    if '$' in xml_content:
        xml_content = re.sub(CORRECT_NESTED_TAG_CLASS_PATTERN, lambda m: f"{m.group(1) or m.group(3)}_{m.group(2) or m.group(4)}", xml_content)

    return xml_content


def parse_xml_to_tree(xml_content):
    # Parse the XML content
    root = ET.fromstring(preprocess_xml_content(xml_content))
    return root


# In XML, the following characters have special meanings and must be escaped to avoid parsing errors:

#     Ampersand (&) → Use &amp;
#     Less than (<) → Use &lt;
#     Greater than (>) → Use &gt;
#     Double quote (") → Use &quot; when inside an attribute value
#     Single quote (') → Use &apos; when inside an attribute value
def decode_special_chars(text):
    return text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&apos;', "'")


def simplify_tree(node, target_box=None):
    """
    Recursively simplify the tree:
    1. Remove nodes with only one child if they do not add any significant info.
    2. Remove unnecessary attributes.
    """
    # Simplify the child nodes first
    children = list(node)
    for child in children:
        simplify_tree(child, target_box)

    # Remove unnecessary attributes from the node
    useful_attributes = ['text', 'resource-id', 'clickable', 'bounds', 'content-desc', 'hint-text', 'tooltip-text']
    if node.tag == 'node' and node.attrib.get('class', ''):
        node.tag = node.attrib['class']
    if node.attrib.get('checkable', 'false') == 'true': useful_attributes.append('checked')
    if node.attrib.get('focusable', 'false') == 'true': useful_attributes.append('focused')
    if node.attrib.get('password', 'false') == 'true': useful_attributes.append('password')

    node.attrib = {key: node.attrib[key] for key in useful_attributes if key in node.attrib and node.attrib[key]}
    #'resource-id' not in node.attrib and
    # Merge the node with its only child if appropriate
    if len(children) == 1 and node.attrib.get('bounds','') != target_box and not node.attrib.get('text', '') and not node.attrib.get('content-desc', '') and all(node.attrib.get(k, 'false') == 'false' for k in ['clickable', 'checked', 'focused', 'password']):
        child = children[0]
        # If the current node and the child node are of the same type and no significant info is lost, merge them
        node.tag = child.tag
        node.attrib.update(child.attrib)
        node[:] = child[:]  # Adopt the children of the child node
        return

    # # If the node has more than one child, ensure it's kept
    # if len(children) > 1:
    #     return


XML_BOX_PATTERN = re.compile(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]')


def tree_to_text(node, level=0, all_boxes=None, skip_statusbar=False):
    """
    Convert the simplified tree to a text format.
    """
    # Recursively add children
    if skip_statusbar and '_statusbar' in node.tag or len(node) == 0 and not node.attrib.get('text', '') and not node.attrib.get('content-desc', '') and all(node.attrib.get(k, 'false') == 'false' for k in ['clickable', 'checked', 'focused', 'password']):
        subtree_str = ''
    else:
        indent = '\t' * level

        node_text = decode_special_chars(node.attrib.get('text',''))
        if len(node_text) == 0:
            node_text = decode_special_chars(node.attrib.get('content-desc',''))

        node_info = "{} text: '{}' ".format(node.tag.split('.')[-1], node_text.replace('\n', ' '))

        # Add additional descriptions
        hint_text = node.attrib.get('hint-text', '').strip()
        if hint_text and hint_text != node_text:
            node_info += f"hint-text: '{hint_text}' "

        tooltip_text = node.attrib.get('tooltip-text', '').strip()
        if tooltip_text and tooltip_text != node_text:
            node_info += f"tooltip-text: '{tooltip_text}' "

        # Add attributes info
        node_info += ', '.join(f"{k}: {v}" for k,v in node.attrib.items() if k not in ['text', 'bounds', 'content-desc', 'hint-text', 'tooltip-text'] and v != 'false')

        # box
        coords = XML_BOX_PATTERN.search(node.attrib.get('bounds',''))
        if coords:
            x1, y1, x2, y2 = map(int, coords.groups())
            box = [x1, y1, x2, y2]
            # box = f'[{x1},{y1},{x2},{y2}]'

            # node_info += f', {box}'
        else: box = None

        if all_boxes is not None:
            all_boxes.append(box)

        child_texts = []
        for child in node:
            child_texts.append(tree_to_text(child, level + 1, all_boxes=all_boxes, skip_statusbar=skip_statusbar))

        subtree_str = f"{indent}{node_info}\n" + ''.join(child_texts)

    return subtree_str


def process_xml(xml_file, target_box=None, resume=True, skip_statusbar=True, skip_lang=False):
    proc_file = xml_file.replace('.xml','_axtree.json')
    if xml_file.endswith(".xml") or xml_file.endswith("_xml.txt"):
        if resume and os.path.exists(proc_file):
            with open(proc_file) as f:
                axtree_info = json.load(f)

            return axtree_info['axtree'].split('\n'), axtree_info['all_boxes']
        else:
            with open(xml_file) as f:
                xml = f.read()
    else: xml = xml_file
    root = parse_xml_to_tree(xml)

    # Step 2: Simplify the tree
    if isinstance(target_box, list):
        target_box = f"[{target_box[0]},{target_box[1]}][{target_box[2]},{target_box[3]}]"
    simplify_tree(root, target_box=target_box)

    # Step 3: Convert the tree to text format
    all_boxes = []
    ax_tree_text = tree_to_text(root, all_boxes=all_boxes, skip_statusbar=skip_statusbar).strip()

    with open(proc_file, "w") as f:
        json.dump({'axtree': ax_tree_text, 'all_boxes': all_boxes}, f, indent=2)

    tree_lines = ax_tree_text.split('\n')

    # Skip non English/Chinese samples
    if skip_lang:
        if detect_invalid_lang(xml):
            all_boxes = []

    return tree_lines, all_boxes


# ---------------------------------------------------------------------------
# find_all_elem_texts_boxes (ported verbatim from tools.py).
# ---------------------------------------------------------------------------
def find_all_elem_texts_boxes(element: Element):
    """
    Recursively find all interactable elements in the XML tree.
    """
    elem_texts_boxes = []

    box = None
    is_leaf = False

    if ('rue' in element.get('clickable','') or
        'rue' in element.get('focusable','') or
        'rue' in element.get('long-clickable','') or
        'rue' in element.get('password','') or
        'rue' in element.get('checkable','')):
        is_interactable = True
    else: is_interactable = False

    box_str = element.get('bounds', None)
    if box_str is not None:
        coords = XML_BOX_PATTERN.search(box_str)
        if coords:
            x1, y1, x2, y2 = map(int, coords.groups())
            box = [x1, y1, x2, y2]

    if len(element) == 0:
        is_leaf = True
    else:
        # Recursively check children
        for child in element:
            elem_texts_boxes.extend(find_all_elem_texts_boxes(child))

    elem_texts_boxes.append({'tag':element.tag.split('.')[-1], 'text':element.attrib.get('text', None), 'box': box, 'package': element.attrib.get('package', None), 'content-desc': element.attrib.get('content-desc', None), 'resource_id': element.attrib.get('resource_id', None), 'is_leaf': is_leaf, 'is_interactable': is_interactable})
    return elem_texts_boxes
