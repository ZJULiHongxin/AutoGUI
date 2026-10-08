"""Pure diff formatting + token counting, ported from WebpageFunctionality/utils/tools.py."""
from __future__ import annotations

from itertools import chain

import tiktoken

encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens


def format_diff(diff, *, diff_limit: int = 250, diff_token_limit: int = 3000, use_additional_prefixes: bool = True) -> tuple:
    # Print the unified diff
    formated_diff = []

    cnt, added_lines, deleted_lines, num_tokens = 0, 0, 0, 0

    for line_id, line in enumerate(diff):
        cnt += 1

        if line_id < diff_limit:
            num_tokens += num_tokens_from_string(line)

        if line_id <= 2: continue

        if line.startswith('+'):
            if line_id < diff_limit and num_tokens < diff_token_limit: formated_diff.append("Added " + line[1:])
            added_lines += 1
        elif line.startswith('-'):
            if line_id < diff_limit and num_tokens < diff_token_limit:
                formated_diff.append("Deleted " + line[1:])
            deleted_lines += 1
        elif len(line) and not line.startswith('@@'):
            if line_id < diff_limit and num_tokens < diff_token_limit: formated_diff.append("Unchanged " + line[1:])

    # Remove the trailing useless elements
    # For example
    # Added		 link 'Events'
    # Unchanged	 button 'Search' hasPopup: menu
    # Unchanged	 button 'Toggle menu open' hasPopup: menu
    # Unchanged		 img 'Profile image'
    # Unchanged	 main '' <-
    idx = len(formated_diff) - 1
    while idx >= 0:
        line = formated_diff[idx]
        if line[line.find("'") + 1] != "'": break
        formated_diff.pop()
        idx -= 1

    if use_additional_prefixes:
        # Check if repositioned or renamed
        for line_id, line in enumerate(formated_diff):
            if not line.startswith("Deleted"): continue
            line_content = line[7:].strip() # Whole node content
            left_quote = line_content.find("'")
            line_elem_type = line_content[:left_quote-1] # node's type (e.g. "StaticText")
            line_text = line_content[left_quote+1:line_content.rfind("'")] # ndde's displayed text
            if len(line_text) == 0: continue # Ignore empty text node

            for check_id in chain(range(max(line_id-10, 0),line_id), range(line_id+1, line_id+11)):
                if check_id >= len(formated_diff): break

                check_line = formated_diff[check_id]
                if check_line.startswith("Added"):

                    check_line_content = check_line[5:].strip() # Whole node content
                    left_quote = check_line_content.find("'")
                    check_line_elem_type = check_line_content[:left_quote-1] # node's type (e.g. "StaticText")
                    check_line_text = check_line_content[left_quote+1:check_line_content.rfind("'")] # ndde's displayed text
                    if len(check_line_text) == 0: continue # Ignore empty text node

                    if check_line_content == line_content:
                        formated_diff[line_id] = formated_diff[line_id].replace("Deleted", "Repositioned {}".format("Upward" if check_id < line_id else "Downward"))
                        formated_diff[check_id] = formated_diff[check_id].replace("Added", "Repositioned Here")
                        break
                    elif check_line_elem_type == line_elem_type:
                        # if len(set(check_line_text.split()).intersection(set(line_text.split()))) > 0 and check_line_text != line_text:
                        #     formated_diff[line_id] = formated_diff[line_id].replace("Deleted", "Before Renaming")
                        #     formated_diff[check_id] = formated_diff[check_id].replace("Added", "After Renaming")
                        #     break
                        # el
                        if check_line_text == line_text:
                            formated_diff[line_id] = formated_diff[line_id].replace("Deleted", "Before Attribute Update")
                            formated_diff[check_id] = formated_diff[check_id].replace("Added", "After Attribute Update")
                            break

    return formated_diff, cnt, added_lines, deleted_lines
