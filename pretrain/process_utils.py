import re

POINT_PATTERN = re.compile(r'<point>\((\d+),(\d+)\)</point>')

def extract_point(text):
    match = POINT_PATTERN.search(text).groups()
    x, y = map(int, match)
    
    return x, y

# is instruction English
def is_english_simple(text):
    try:
        text.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


# bbox -> point (str)
def bbox_2_point(bbox, scale=100):
    # bbox [left, top, right, bottom]
    if scale > 1:
        point = [(bbox[0]+bbox[2])/2*scale, (bbox[1]+bbox[3])/2*scale]
        point = [f"{int(item):d}" for item in point]
    else:
        point = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        point = [f"{item:.2f}" for item in point]
    point_str = "({},{})".format(point[0], point[1])
    return point_str

# bbox -> bbox (str)
def bbox_2_bbox(bbox, scale=100):
    if scale > 1:
        bbox = [f"{int(item*scale):d}" for item in bbox]
    else:
        bbox = [f"{item:.2f}" for item in bbox]
    bbox_str = "({},{},{},{})".format(bbox[0], bbox[1], bbox[2], bbox[3])
    return bbox_str

# point (str) -> point
def pred_2_point_old(s):
    floats = re.findall(r'-?\d+\.?\d*', s)
    floats = [float(num/1000) for num in floats]
    if len(floats) == 2:
        click_point = floats
    elif len(floats) == 4:
        click_point = [(floats[0]+floats[2])/2, (floats[1]+floats[3])/2]
    return click_point

# bbox (qwen str) -> bbox
def extract_bbox(s):
    # Regular expression to find the content inside <box> and </box>
    pattern = r"<box>\((\d+,\d+)\),\((\d+,\d+)\)</box>"
    matches = re.findall(pattern, s)
    # Convert the tuples of strings into tuples of integers
    return [(int(x.split(',')[0]), int(x.split(',')[1])) for x in sum(matches, ())]


# bbox (qwen str) -> bbox
SEECLICK_BOX_PATTERN = re.compile(r"\((\d+,\d+)\),\((\d+,\d+)\)")
def extract_bbox(pred):
    # Regular expression to find the content inside <box> and </box>
    matches = SEECLICK_BOX_PATTERN.findall(pred)
    # Convert the tuples of strings into tuples of integers
    
    try:
        points = []
        
        for point in matches[-1]:
            x, y = point.split(',')
            points.extend([int(x), int(y)])
    except:
        points = None

    return points

# point (str) -> point
BRACKET_COORD_PATTERN = re.compile(r'\[(.*?)\]')
GENERAL_COORD_PATTERN = re.compile(r'-?\d+\.?\d*')


def pred_2_point(pred, keep_box=True, scale=1000):
    click_point = None
    if '[[' in pred: # For CogAgent
        coords_start = pred.find('[[')
        if coords_start != -1:
            coords_end = pred.find(']]')
            if coords_end != -1:
                coords_str = pred[coords_start+2:coords_end]
                try:
                    # The bounding box coordinates in the CogAgent's output use the format [[x1, y1, x2, y2]], with the origin at the top left corner, the x-axis to the right, and the y-axis downward. (x1, y1) and (x2, y2) are the top-left and bottom-right corners, respectively, with values as relative coordinates multiplied by 1000 (prefixed with zeros to three digits).
                    click_point = [x / scale for x in map(float, coords_str.split(','))]
                except:
                    raise ValueError("Cannot extract click point from {}".format(pred))
    elif '[' in pred:
        matches = [(match.group(), (match.start(), match.end())) for match in BRACKET_COORD_PATTERN.finditer(pred)]

        if matches:
            # We take the last one
            last_valid_match_id = len(matches) - 1
            while last_valid_match_id >=0:
                click_point_str, start_end = matches[last_valid_match_id]
                try:
                    click_point = list(map(float, click_point_str[1:-1].split(',')))
                    break
                except: pass
                last_valid_match_id -= 1
            else:
                raise ValueError("Cannot extract click point from {}".format(pred))

            # If there are two coordinates enclosed with brackets and they are different and their appearances in the response are not far away, they may be represent the top-left and bottom-right corners, respectively.
            if len(click_point) == 2 and last_valid_match_id > 0 and (start_end[0] - matches[last_valid_match_id-1][1][1]) < 30:
                try:
                    another_point = list(map(float, matches[last_valid_match_id-1][0][1:-1].split(', ')))
                    if len(another_point) == 2:
                        click_point = [(another_point[0] + click_point[0]) / 2, (another_point[1] + click_point[1]) / 2]
                except: pass
    elif pred.startswith('<box>'): # '<box>598 102 673 406</box>.'
        coords = re.findall(r'\d+', pred)

        # Convert to integers
        click_point = [int(num) for num in coords]
    
    if click_point is None: # For SeeClick
        if '<box>' in pred: # For QWen-VL-Chat
            click_point = extract_bbox(pred)
        else:
            floats = GENERAL_COORD_PATTERN.findall(pred)
            
            if floats:
                click_point = []
                for num in floats:
                    try:
                        num = float(num)
                        click_point.append(num)
                    except: pass
        
    assert click_point is not None, "Cannot extract click point from {}".format(pred)
    assert len(click_point) in [2,4], "Invalid click point {} found in {}".format(click_point, pred)
    
    if not keep_box and len(click_point) == 4:
        click_point = [(click_point[0]+click_point[2])/2, (click_point[1]+click_point[3])/2]

    # In case where the coordinates are normalized in the range [0, 1000)
    if any(x > 1 for x in click_point):
        click_point = [x / scale for x in click_point]

    return click_point

