"""Web accessibility-tree (AXTree) parsing -> text-tree.

Ported verbatim from the web half of
``WebpageFunctionality/utils/process_axtree.py``: ``NUMBERING_PATTERN``,
``prune_accessibility_tree_wo_bound``, ``parse_accessibility_tree``,
``clean_accessibility_tree`` (spelling corrected from the source's
``clean_accesibility_tree``), and ``process_axtree``.

Only the pure tree-parsing functions are included. The async Playwright/CDP
scraper ``extract_axtree`` (and its helpers ``get_bounding_client_rect_async``
/ ``process_node``) is deliberately dropped (spec deviation #4): no network,
async, or browser-automation code is pulled in, and no hostnames/IPs/endpoints
are carried across. ``MARKER_STR`` is imported from the shared pruning module
rather than re-declared.
"""

import json
import os
import re
from typing import Any

from autogui_anno.axtree.prune import MARKER_STR

IN_VIEWPORT_RATIO_THRESHOLD = 0.6
INVALID_NODE_ROLES = ["generic", "img", "list", "strong", "paragraph", "banner", "navigation", "Section", "LabelText", "Legend", "listitem", "alert", "superscript", "LineBreak", "Canvas"]
IGNORED_ACTREE_PROPERTIES = (
    "focusable",
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
    "disabled",
    "describedby",
    "roledescription"
)

NUMBERING_PATTERN = re.compile(r'\[\d+\]')


def prune_accessibility_tree_wo_bound(
    accessibility_tree,
) -> tuple[str, dict[str, Any]]:
    """Parse the accessibility tree into a string text"""

    def remove_node_in_graph(node) -> None:
        # update the node information in the accessibility tree
        nodeid = node["nodeId"]
        parent_nodeid = node["parentId"]

        # If the node's parent is not in the accessibility tree, remove it. This case happen when the AXtree only contains the nodes inside the viewport.
        if parent_nodeid not in accessibility_tree:
            accessibility_tree[nodeid]["parentId"] = "[REMOVED]"
            return

        children_nodeids = node["childIds"]
        # update the children of the parent node
        assert (
            accessibility_tree[parent_nodeid].get("parentId", "Root")
            is not None
        )
        # remove the nodeid from parent's childIds
        try:
            index = accessibility_tree[parent_nodeid]["childIds"].index(
                nodeid
            )
            accessibility_tree[parent_nodeid]["childIds"].pop(index)
        except:
            index = len(accessibility_tree[parent_nodeid]["childIds"])

        # Insert children_nodeids in the same location
        for child_nodeid in children_nodeids:
            accessibility_tree[parent_nodeid]["childIds"].insert(
                index, child_nodeid
            )
            index += 1
        # update children node's parent
        for child_nodeid in children_nodeids:
            if child_nodeid not in accessibility_tree: continue
            accessibility_tree[child_nodeid][
                "parentId"
            ] = parent_nodeid
        # mark as removed
        accessibility_tree[nodeid]["parentId"] = "[REMOVED]"

    for obs_node_id, node in accessibility_tree.items():
        valid_node = True
        try:
            role = node["role"]["value"]
            name = node["name"]["value"]

            node_str = f"[{obs_node_id}] {role} {repr(name)}"
            properties = []

            if role == 'textbox':
                for x in node["name"]['sources']:
                    if x['type'] == 'placeholder' and 'value' in x.keys():
                        properties.append(f"placeholder: [{x['value']['value']}]")

            for property in node.get("properties", []):
                try:
                    if property["name"] in IGNORED_ACTREE_PROPERTIES:
                        continue
                    properties.append(
                        f'{property["name"]}: {property["value"]["value"]}'
                    )
                except KeyError:
                    pass

            if properties:
                node_str += " " + " ".join(properties)

            # check valid
            if not node_str.strip():
                valid_node = False

            # empty generic node
            if not name.strip():
                if not properties:
                    if role in INVALID_NODE_ROLES:
                        valid_node = False
                elif role in ["listitem"]:
                    valid_node = False

            if not valid_node:
                remove_node_in_graph(node)
                continue
        except Exception as e:
            valid_node = False
            remove_node_in_graph(node)

    for nodeId in list(accessibility_tree.keys()):
        if accessibility_tree[nodeId].get("parentId", "-1") == "[REMOVED]":
            del accessibility_tree[nodeId]

    return accessibility_tree


def parse_accessibility_tree(accessibility_tree, start_node_id: str = '1', numbering_start: int = 1) -> tuple[str, dict[str, Any]]:
    """Parse the accessibility tree into a string text."""
    obs_nodes_info = {}
    reorder = {}  # map numbering orders to real identifiers
    node_ids = set(accessibility_tree.keys())

    def is_valid_node(node, role, name, properties):
        if not name.strip() and not properties and role in INVALID_NODE_ROLES:
            return False
        if role == "listitem" and not properties:
            return False
        return True

    # Find the root node
    for start_node_id, v in accessibility_tree.items():
        if v['role']['value'] == 'RootWebArea': break
    else: raise Exception("Invalid webpage without a root node!")

    stack = [(start_node_id, 0)]
    tree_lines = []

    while stack:
        obs_node_id, depth = stack.pop()
        if obs_node_id not in node_ids:
            continue

        node = accessibility_tree[obs_node_id]
        indent = "\t" * depth

        role = node["role"].get("value", None)
        if role is None: continue

        name = node["name"].get("value", None)
        if name is None: continue

        if node.get("backendDOMNodeId", None) is None: continue

        reorder[str(numbering_start)] = obs_node_id

        properties = []

        if role == 'textbox':
            for x in node["name"]['sources']:
                if x['type'] == 'placeholder' and 'value' in x.keys():
                    properties.append(f"placeholder: [{x['value']['value']}]")
                    break

        for property in node.get("properties", []):
                try:
                    if property["name"] in IGNORED_ACTREE_PROPERTIES:
                        continue
                    properties.append(
                        f'{property["name"]}: {property["value"]["value"]}'
                    )
                except KeyError:
                    pass

        node_str = f"{role} {repr(name)}" if numbering_start != -1 else f"{role} {repr(name)}"

        if properties:
            node_str += " " + " ".join(properties)

        numbering_start += 1
        valid_node = is_valid_node(node, role, name, properties)

        if valid_node:
            tree_lines.append(f"{indent}{node_str}")
            obs_nodes_info[obs_node_id] = {
                "nodeId": obs_node_id,
                "backend_id": node["backendDOMNodeId"],
                "union_bound": node["union_bound"],
                "text": node_str,
                "name": name,
                "role": role,
                'parentId': node.get('parentId', ''),
                'childIds': node['childIds']
            }

        for child_node_id in reversed(node["childIds"]):
            child_depth = depth + 1 if valid_node else depth
            stack.append((child_node_id, child_depth))

    # tree_str = "\n".join(tree_lines)
    if len(obs_nodes_info) == 0:
            print("Empty AXTree")
    # update child IDs
    return tree_lines, obs_nodes_info, reorder


STATICTEXT_PATTERN = re.compile(r"StaticText (.+)") # The axtree does not contain ID markers
# Raw: r"\[\d+\] StaticText (.+)"


def clean_accessibility_tree(tree_lines: list) -> list:
    """further clean accesibility tree"""
    clean_lines: list[str] = []
    for idx, line in enumerate(tree_lines):
        # remove statictext if the content already appears in the previous line
        if "statictext" in line.lower():
            prev_lines = clean_lines[-3:]

            # match = STATICTEXT_PATTERN.search(line, re.DOTALL)
            if 'StaticText ' in line:
                text_start = line.find("'")
                text_end = line.find("'", text_start+1)
                static_text = line[text_start:text_end]

                # static_text = match.group(1)[1:-1]  # remove the quotes

                if static_text and all(
                    static_text not in prev_line
                    for prev_line in prev_lines
                ) or MARKER_STR in static_text:
                    clean_lines.append(line)
        else:
            clean_lines.append(line)

    return clean_lines


def process_axtree(axtree_file, *, resume=False, node_list=None):
    clean_axtree_with_markers_file = axtree_file.replace(".txt", "_clean.json")
    if resume and os.path.exists(clean_axtree_with_markers_file):
        with open(clean_axtree_with_markers_file, "r") as f:
            axtree_info = json.load(f)

        print("Resume from", clean_axtree_with_markers_file)
        return axtree_info["clean_axtree"], axtree_info["invalid_markers"]

    with open(axtree_file, 'r') as f:
        axtree_list = json.load(f)

    # Convert the Axtree to dict and record aria-ids
    axtree, invalid_markers = {}, []
    for node in axtree_list:
        axtree[node["nodeId"]] = node

        # for prop in node.get('properties', []):
        #     if prop["name"] == 'roledescription':
        #         aria_attrs = json.loads(prop["value"]["value"])
        #         if "id" in aria_attrs:
        #             ariaid2nodeid[aria_attrs["id"]] = node["nodeId"]

    # Label each tabbable candidates
    if node_list is not None:
        for node_info in node_list:
            if 'name' not in node_info["axtree_node"]: continue

            node_id, node_marker, node_text, node_href = node_info["axtree_node"]["nodeId"], node_info["hint_marker_text"], node_info["axtree_node"].get("value", node_info["axtree_node"]["name"]), node_info.get("href", '')

            if node_id not in axtree:
                invalid_markers.append({'node_id': node_id, 'marker': node_marker})
                continue

            # node_name_sources = axtree[node_id]["name"]["sources"]

            axtree[node_id]["name"]["value"] = f'Marker:[{node_marker}]{axtree[node_id]["name"]["value"]}'
            # for name_source in node_name_sources:
            #     if name_source.get("attribute", "") == "aria-label":
            #         new_str = f'Marker:[{node_marker}]{name_source["value"]["value"]}'
            #         name_source["value"]["value"] = name_source["attributeValue"]["value"] = new_str
            #         break

    axtree = prune_accessibility_tree_wo_bound(axtree)

    # 生成不带[x]标签的AXTree，以提高效率
    tree_lines, obs_nodes_info, reorder = parse_accessibility_tree(
            axtree
        )

    tree_lines = clean_accessibility_tree(tree_lines)

    # get the axtree_json
    with open(clean_axtree_with_markers_file, "w") as f:
        json.dump({
            'clean_axtree': tree_lines,
            'invalid_markers': invalid_markers
        }, f, indent=2)

    return tree_lines, invalid_markers
