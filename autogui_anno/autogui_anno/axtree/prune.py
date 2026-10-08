"""Pure AXTree pruning helpers.

Ported verbatim from WebpageFunctionality/utils/tools.py, with the module
globals MENUITEM_LIMIT / TAB_LIMIT / TEXT_MAX_LEN converted into function
kwargs. No logic changes beyond that threading. Pure module: no imports from
the LLM/network layers.
"""
import re

CLICKABLE = [
    "button",
    "link",
    "checkbox",
    "radio",
    "menuitem",
    "tab",
    "combobox",
    "listbox",
    "slider",
    "textbox"
]

MARKER_STR = 'Marker:['

USELESS_ELEM_TYPES = ["'advertisement'", "status ''", "generic ''", "gridcell ''", "deletion ''", "alert ''", "superscript ''", "paragraph ''", "heading ''", "link ''","linebreak", "canvas ''", "option ''", "graphics-symbol ''"]

# (parent element, child, 0 means keeping the former 1 means keeping the latter)
REDUNDANT_PAIR = [["generic", "link", 1], ["heading", "link", 1], ["generic", "button", 1], ["link", "img", 0], ["LabelText", "radio", 1], ["figure", "img", 1], ['link', 'StaticText', 0]]

MERGED_PAIRS = [['link', 'StaticText'], ['link', 'image'], ['button', 'StaticText']]

USELESS_ATTR = ["describedby:"]

REMOVE = [" disabled: True", "required: False"]

LABEL_PATTERN = re.compile(r'\[\d+\]')
LABEL_PATTERN_WO_BRACKETS = re.compile(r'\[([^\]]+)\]')


def remove_labels(content: str):
    return LABEL_PATTERN.sub("", content)


def extract_labels(content: str):
    return LABEL_PATTERN_WO_BRACKETS.findall(content)


def get_markers(lines: list):
    marker_line_idx = {}
    for i in range(len(lines)):
        line = lines[i]
        first_marker_start = line.find(MARKER_STR)

        marker_start = first_marker_start
        if marker_start == -1: continue
        recorded = False

        while True:
            marker_end = line.find("]", marker_start)

            marker = line[marker_start + len(MARKER_STR): marker_end]

            if not recorded:
                marker_line_idx[marker] = i # We only record the first (i.e. the most updated) marker to avoid a situation where several marker point to the same line
                recorded = True

            marker_start = line.find(MARKER_STR, marker_end)

            if marker_start == -1:
                lines[i] = line[:first_marker_start] + line[marker_end+1:]
                break

    return marker_line_idx


def count_tab(line: str):
    if len(line) == 0: return 0

    line_tab_cnt = 0
    tab_idx = line.find('\t')
    while line[tab_idx] == '\t':
        line_tab_cnt += 1
        tab_idx += 1

    return line_tab_cnt


def prune_schema_text(schema_lines: list, *, text_max_len: int = 122):
    i = 0

    new_schema_lines = []
    while i < len(schema_lines):
        line = schema_lines[i]
        # Truncate elem. text
        left_quote, right_quote = line.find('"'), line.rfind('"')

        if right_quote != -1:
            text = line[left_quote+1:right_quote+1]


            if len(text) > text_max_len:
                split_id = text.find('. ')
                split_id = split_id + 1 if split_id != -1 else text_max_len

                unicode_id = text.find('\\', split_id-6, min(text_max_len, split_id+4)) # -6 because \\uXXXX
                if unicode_id != -1:
                    split_id = unicode_id

                new_line = line[:left_quote+1+split_id] + '"'
            else:
                new_line = line[:right_quote+1]
        else:
            new_line = re.sub(r'\d+', '', line).rstrip()

        new_schema_lines.append(new_line)
        i += 1

    return new_schema_lines


def prune_static_text(static_text_lines: list, *, line_limit: int = 9999, remove_hidden: bool = False, only_remove_attrs: bool = False, with_markers: bool = False, menuitem_limit: int = 3, tab_limit: int = 10, text_max_len: int = 122):
    markers = {}

    if not only_remove_attrs:
        i = 0
        # Pre-process
        while i < min(len(static_text_lines), line_limit + 100):
            line = static_text_lines[i]

            # Skip the line with tabbability marker
            if with_markers and MARKER_STR in line:
                i += 1
                continue

            lower_line = line.lower()
            # remove useless elements
            is_useless = False
            for useless_elem in USELESS_ELEM_TYPES:
                if useless_elem in lower_line:
                    del static_text_lines[i]
                    is_useless = True; break

            if not is_useless: i+= 1

        i = 0
        # Start cleaning redundant lines
        while i < min(len(static_text_lines), line_limit):
            line = static_text_lines[i]

            # Skip the line with tabbability marker
            if with_markers and MARKER_STR in line:
                i += 1
                continue

            # remove hidden elements
            if remove_hidden and "hidden: True" in line:
                del static_text_lines[i]
                continue

            # remove redundant menu items
            if "menuitem" in line:
                menuitem_cnt = 1
                while i + menuitem_cnt < len(static_text_lines) and "menuitem" in static_text_lines[i+menuitem_cnt]:
                    menuitem_cnt += 1

                if menuitem_cnt > menuitem_limit:
                    cnt_to_delete = menuitem_cnt - menuitem_limit
                    idx_to_delete = i + menuitem_limit
                    while cnt_to_delete > 0 and idx_to_delete < len(static_text_lines):
                        if with_markers and 'Marker:[' in line: idx_to_delete += 1
                        else:
                            del static_text_lines[idx_to_delete]
                        cnt_to_delete -= 1
                    i = idx_to_delete
                else:
                    i = i + menuitem_limit
            elif "row" in line or "tablist" in line or "region 'Map'" in line:
                is_map = "region 'Map'" in line
                is_tablist = "tablist" in line

                menuitem_cnt = 0

                # Count #tab
                rowline_tab_cnt = 0
                tab_idx = line.find('\t')
                while line[tab_idx] == '\t':
                    rowline_tab_cnt += 1
                    tab_idx += 1

                i += 1
                while i < len(static_text_lines):
                    this_line = static_text_lines[i]

                    thisline_tab_cnt = this_line[:this_line.find(' ')].count('\t')

                    if thisline_tab_cnt <= rowline_tab_cnt or (is_map and 'button ' not in this_line):
                        break # Detect the end of the listed elements

                    if menuitem_cnt >= (tab_limit if is_tablist else menuitem_limit):
                        if not (with_markers and MARKER_STR in line):
                            del static_text_lines[i]
                    else:
                        if i + 1 < len(static_text_lines):
                            next_line = static_text_lines[i+1]
                            nextline_tab_cnt = next_line[:next_line.find(' ')].count('\t')

                            if nextline_tab_cnt > thisline_tab_cnt:
                                nextline_text = next_line[next_line.find("'"):next_line.rfind("'")]

                                if not (with_markers and MARKER_STR in line):
                                    thisline_text = this_line[this_line.find("'"):this_line.rfind("'")]
                                    if thisline_text == nextline_text:
                                        del static_text_lines[i + 1]

                        menuitem_cnt += 1

                        # Go to the next menu item
                        while i < len(static_text_lines):
                            i += 1
                            if i >= len(static_text_lines): break
                            iter_line = static_text_lines[i]
                            nextline_tab_cnt = iter_line[:iter_line.find(' ')].count('\t')

                            # Detect the next menu item or the end of the row
                            if nextline_tab_cnt <= rowline_tab_cnt + 1: break
            elif "SvgRoot ''" in line:
                line_tab_cnt = count_tab(line)

                while True:
                    if not (with_markers and MARKER_STR in line):
                        del static_text_lines[i]
                    else: i += 1

                    if i >= len(static_text_lines): break

                    next_line = static_text_lines[i]
                    next_line_tab_cnt = count_tab(next_line)

                    if next_line_tab_cnt <= line_tab_cnt:
                        break
            else:
                i += 1

        idx_to_delete = line_limit
        while idx_to_delete < len(static_text_lines):
            if not (with_markers and MARKER_STR in static_text_lines[idx_to_delete]):
                del static_text_lines[idx_to_delete]

            idx_to_delete += 1

    # Post-process
    # Extract tabability markers. The lines will be modified in-place.
    markers = get_markers(static_text_lines) if with_markers else {}

    # Stage 1: Condense duplicate element pairs. If the two elements in the pair have the same displayed strings, only one of them will be retained.
    i = 0
    while i < len(static_text_lines) - 1:
        line = static_text_lines[i]
        line_tab_cnt = count_tab(line)
        line_text = line[line.find("'"):line.rfind("'")]

        for redundant_pair in REDUNDANT_PAIR:
            if redundant_pair[0] in line:
                # If the first element is to be retained, iterate over its first-level child nodes
                if redundant_pair[-1] == 0:
                    j = i + 1
                    while j < len(static_text_lines):
                        next_line = static_text_lines[j]
                        next_line_tab_cnt = count_tab(next_line)

                        if next_line_tab_cnt <= line_tab_cnt: break

                        if '\t' * (line_tab_cnt + 1) + f'{redundant_pair[1]}' in next_line:
                            next_line_text = next_line[next_line.find("'"):next_line.rfind("'")]
                            if next_line_text.strip() == line_text.strip() and not (with_markers and MARKER_STR in next_line):
                                del static_text_lines[j]

                                # Adjust the indices of the lines pointed by the markers
                                for k in markers:
                                    if markers[k] == j: markers[k] = i
                                    if markers[k] > j: markers[k] -= 1

                        j += 1

                # If the second element is to be retained, remove the first one the adjust the level of the 2nd one.
                elif redundant_pair[-1] == 1 and not (with_markers and MARKER_STR in line):
                    next_line = static_text_lines[i+1]
                    next_line_text = next_line[next_line.find("'"):next_line.rfind("'")]
                    if next_line_text.strip() == line_text.strip():
                        static_text_lines[i+1] = static_text_lines[i+1][1:] # adjust it up a level by deleting a \t
                        del static_text_lines[i]

                        # Adjust the indices of the lines pointed by the markers
                        for k in markers:
                            if markers[k] > i: markers[k] -= 1

                        i -= 1
                        break
        i += 1

    # Stage 2: Merging elements. The first element which has no displayed text in the pair will inherit the text of the second one and then the second one will be deleted.
    i = 0
    while i < min(len(static_text_lines) - 1, line_limit + 100):
        line = static_text_lines[i]
        next_line = static_text_lines[i+1]

        for pair in MERGED_PAIRS:
            if f"{pair[0]} '" in line and f"{pair[1]} '" in next_line:
                line_text = line[line.find("'")+1:line.rfind("'")]
                next_line_text = next_line[next_line.find("'")+1:next_line.rfind("'")]
                if len(line_text) == 0:
                    line_tab_cnt = count_tab(line)
                    next_line_tab_cnt = count_tab(next_line)

                    if next_line_tab_cnt >= line_tab_cnt:
                        static_text_lines[i] = line.replace("''", next_line[next_line.find("'"):next_line.rfind("'")+1])
                        del static_text_lines[i+1]
                        # modify marker line indices
                        for k in markers:
                            if markers[k] > i: markers[k] -= 1
                break

        i += 1

    # Stage 3: Truncate lengthy texts and remove redundant attributes.
    i = 0
    while i < len(static_text_lines):
        # remove useless attrs
        line = static_text_lines[i]

        for useless_attr in USELESS_ATTR:
            attr_id = line.find(useless_attr)
            if attr_id == -1: continue

            attr_value_id = attr_id + len(useless_attr)
            while attr_value_id < len(line) and line[attr_value_id] == ' ':
                attr_value_id += 1

            next_attr_id = line.find(':', attr_value_id)
            if next_attr_id == -1: static_text_lines[i] = line[:attr_id-1]
            else:
                attr_value_end = line.rfind(' ', attr_value_id, next_attr_id)
                static_text_lines[i] = line[:attr_id-1] + line[attr_value_end:]

        # Remove useless attributes
        for substr in REMOVE:
            static_text_lines[i] = static_text_lines[i].replace(substr, "")

        # Truncate elem. text
        text_start, right_quote = line.find("'") + 1, line.rfind("'")

        # Skip all markers
        marker_idx = line.find(MARKER_STR, text_start)
        while True:
            next_marker_idx = line.find(MARKER_STR, marker_idx + 1)
            if next_marker_idx == -1:
                text_start = line.find(']', marker_idx) + 1
                break

            marker_idx = next_marker_idx

        # Truncate the real text
        text = line[text_start:right_quote]

        if len(text) > text_max_len:
            split_id = text.find('. ')
            split_id = split_id + 1 if split_id != -1 else text_max_len

            unicode_id = text.find('\\', split_id-6, min(text_max_len, split_id+4)) # -6 because \\uXXXX
            if unicode_id != -1:
                split_id = unicode_id

            static_text_lines[i] = line[:text_start + split_id + 1] + line[right_quote:]

        i += 1

    return markers
