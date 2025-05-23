'''
Adapted from https://github.com/google-research/google-research/tree/master/android_in_the_wild
'''

import jax
import jax.numpy as jnp
import numpy as np

'''
Adapted from https://github.com/google-research/google-research/tree/master/android_in_the_wild
'''

import enum

action_id2text = {
  0: "swipe down",
  1: "swipe up",
  2: "select",
  3: "type text",
  4: "click",
  5: "PRESS_BACK",
  6: "PRESS_HOME",
  7: "PRESS_ENTER",
  8: "swipe left",
  9: "swipe right",
  10: "STATUS_TASK_COMPLETE",
  11: "STATUS_TASK_IMPOSSIBLE"
} 
class ActionType(enum.IntEnum):

  # Placeholders for unused enum values
  UNUSED_0 = 0
  UNUSED_1 = 1
  UNUSED_2 = 2
  UNUSED_8 = 8
  UNUSED_9 = 9

  ########### Agent actions ###########

  # A type action that sends text to the emulator. Note that this simply sends
  # text and does not perform any clicks for element focus or enter presses for
  # submitting text.
  TYPE = 3

  # The dual point action used to represent all gestures.
  DUAL_POINT = 4

  # These actions differentiate pressing the home and back button from touches.
  # They represent explicit presses of back and home performed using ADB.
  PRESS_BACK = 5
  PRESS_HOME = 6

  # An action representing that ADB command for hitting enter was performed.
  PRESS_ENTER = 7

  ########### Episode status actions ###########

  # An action used to indicate the desired task has been completed and resets
  # the environment. This action should also be used in the case that the task
  # has already been completed and there is nothing to do.
  # e.g. The task is to turn on the Wi-Fi when it is already on
  STATUS_TASK_COMPLETE = 10

  # An action used to indicate that desired task is impossible to complete and
  # resets the environment. This can be a result of many different things
  # including UI changes, Android version differences, etc.
  STATUS_TASK_IMPOSSIBLE = 11


_TAP_DISTANCE_THRESHOLD = 0.14  # Fraction of the screen
ANNOTATION_WIDTH_AUGMENT_FRACTION = 1.4
ANNOTATION_HEIGHT_AUGMENT_FRACTION = 1.4

# Interval determining if an action is a tap or a swipe.
_SWIPE_DISTANCE_THRESHOLD = 0.04


def _yx_in_bounding_boxes(
    yx, bounding_boxes
):
  """Check if the (y,x) point is contained in each bounding box.

  Args:
    yx: The (y, x) coordinate in pixels of the point.
    bounding_boxes: A 2D int array of shape (num_bboxes, 4), where each row
      represents a bounding box: (y_top_left, x_top_left, box_height,
      box_width). Note: containment is inclusive of the bounding box edges.

  Returns:
    is_inside: A 1D bool array where each element specifies if the point is
      contained within the respective box.
  """
  y, x = yx

  # `bounding_boxes` has shape (n_elements, 4); we extract each array along the
  # last axis into shape (n_elements, 1), then squeeze unneeded dimension.
  top, left, height, width = [
      jnp.squeeze(v, axis=-1) for v in jnp.split(bounding_boxes, 4, axis=-1)
  ]

  # The y-axis is inverted for AndroidEnv, so bottom = top + height.
  bottom, right = top + height, left + width

  return jnp.logical_and(y >= top, y <= bottom) & jnp.logical_and(
      x >= left, x <= right)


def _resize_annotation_bounding_boxes(
    annotation_positions, annotation_width_augment_fraction,
    annotation_height_augment_fraction):
  """Resize the bounding boxes by the given fractions.

  Args:
    annotation_positions: Array of shape (N, 4), where each row represents the
      (y, x, height, width) of the bounding boxes.
    annotation_width_augment_fraction: The fraction to augment the box widths,
      E.g., 1.4 == 240% total increase.
    annotation_height_augment_fraction: Same as described for width, but for box
      height.

  Returns:
    Resized bounding box.

  """
  height_change = (
      annotation_height_augment_fraction * annotation_positions[:, 2])
  width_change = (
      annotation_width_augment_fraction * annotation_positions[:, 3])

  # Limit bounding box positions to the screen.
  resized_annotations = jnp.stack([
      jnp.maximum(0, annotation_positions[:, 0] - (height_change / 2)),
      jnp.maximum(0, annotation_positions[:, 1] - (width_change / 2)),
      jnp.minimum(1, annotation_positions[:, 2] + height_change),
      jnp.minimum(1, annotation_positions[:, 3] + width_change),
  ],
                                  axis=1)
  return resized_annotations


def is_tap_action(normalized_start_yx,
                  normalized_end_yx):
  distance = jnp.linalg.norm(
      jnp.array(normalized_start_yx) - jnp.array(normalized_end_yx))
  return distance <= _SWIPE_DISTANCE_THRESHOLD


def _is_non_dual_point_action(action_type):
  return jnp.not_equal(action_type, ActionType.DUAL_POINT)


def _check_tap_actions_match(
    tap_1_yx,
    tap_2_yx,
    annotation_positions,
    matching_tap_distance_threshold_screen_percentage,
    annotation_width_augment_fraction,
    annotation_height_augment_fraction,
):
  """Determines if two tap actions are the same."""
  both_in_box = False
  
  if len(annotation_positions):
    resized_annotation_positions = _resize_annotation_bounding_boxes(
        annotation_positions,
        annotation_width_augment_fraction,
        annotation_height_augment_fraction,
    )

    # Check if the ground truth tap action falls in an annotation's bounding box.
    tap1_in_box = _yx_in_bounding_boxes(tap_1_yx, resized_annotation_positions)
    tap2_in_box = _yx_in_bounding_boxes(tap_2_yx, resized_annotation_positions)
    both_in_box = jnp.max(tap1_in_box & tap2_in_box)

  # If the ground-truth tap action falls outside any of the annotation
  # bounding boxes or one of the actions is inside a bounding box and the other
  # is outside bounding box or vice versa, compare the points using Euclidean
  # distance.
  within_threshold = (
      jnp.linalg.norm(jnp.array(tap_1_yx) - jnp.array(tap_2_yx))
      <= matching_tap_distance_threshold_screen_percentage
  )
  return jnp.logical_or(both_in_box, within_threshold)


def _check_drag_actions_match(
    drag_1_touch_yx,
    drag_1_lift_yx,
    drag_2_touch_yx,
    drag_2_lift_yx,
):
  """Determines if two drag actions are the same."""
  # Store drag deltas (the change in the y and x coordinates from touch to
  # lift), magnitudes, and the index of the main axis, which is the axis with
  # the greatest change in coordinate value (e.g. a drag starting at (0, 0) and
  # ending at (0.3, 0.5) has a main axis index of 1).
  drag_1_deltas = drag_1_lift_yx - drag_1_touch_yx
  drag_1_magnitudes = jnp.abs(drag_1_deltas)
  drag_1_main_axis = np.argmax(drag_1_magnitudes)
  drag_2_deltas = drag_2_lift_yx - drag_2_touch_yx
  drag_2_magnitudes = jnp.abs(drag_2_deltas)
  drag_2_main_axis = np.argmax(drag_2_magnitudes)

  return jnp.equal(drag_1_main_axis, drag_2_main_axis)


def check_actions_match(
    ref_action_type,
    ref_action_attr,
    pred_action_type,
    pred_action_attr,
    annotation_positions,
    tap_distance_threshold = _TAP_DISTANCE_THRESHOLD,
    annotation_width_augment_fraction = ANNOTATION_WIDTH_AUGMENT_FRACTION,
    annotation_height_augment_fraction = ANNOTATION_HEIGHT_AUGMENT_FRACTION,
):
    """Determines if two actions are considered to be the same.

    Two actions being "the same" is defined here as two actions that would result
    in a similar screen state.

    Args:
        action_1_touch_yx: The (y, x) coordinates of the first action's touch.
        action_1_lift_yx: The (y, x) coordinates of the first action's lift.
        action_1_action_type: The action type of the first action.
        action_2_touch_yx: The (y, x) coordinates of the second action's touch.
        action_2_lift_yx: The (y, x) coordinates of the second action's lift.
        action_2_action_type: The action type of the second action.
        annotation_positions: The positions of the UI annotations for the screen. It
        is A 2D int array of shape (num_bboxes, 4), where each row represents a
        bounding box: (y_top_left, x_top_left, box_height, box_width). Note that
        containment is inclusive of the bounding box edges.
        tap_distance_threshold: The threshold that determines if two taps result in
        a matching screen state if they don't fall the same bounding boxes.
        annotation_width_augment_fraction: The fraction to increase the width of the
        bounding box by.
        annotation_height_augment_fraction: The fraction to increase the height of
        of the bounding box by.

    Returns:
        A boolean representing whether the two given actions are the same or not.
    """
    if ref_action_type != pred_action_type: return [ref_action_type, False, False] # gt action type, action type match, action match
    
    if ref_action_type == 'click':
        taps_match = _check_tap_actions_match(
            ref_action_attr['target'],
            pred_action_attr['target'],
            annotation_positions,
            tap_distance_threshold,
            annotation_width_augment_fraction,
            annotation_height_augment_fraction,
        )
        return [ref_action_type, True, taps_match]
    elif ref_action_type == 'input_text':
        gt_text, pred_text = ref_action_attr['text'].lower(), pred_action_attr['text'].lower() # AITW先把字符串lower再匹配
        text_match =  (gt_text == pred_text) or (
                            gt_text in pred_text) or (
                            pred_text in gt_text)
        
        return [ref_action_type, True, text_match]
    elif ref_action_type == 'swipe':
        direction_match = ref_action_attr['direction'] == pred_action_attr['direction']
        return [ref_action_type, True, direction_match]
    elif ref_action_type == 'status':
        # answer_match = ref_action_attr['answer'] in pred_action_attr['answer'] or pred_action_attr['answer'] in ref_action_attr['answer']
        status_match = ref_action_attr['goal_status'] == pred_action_attr['goal_status']
        return [ref_action_type, True, status_match]
    else:
        return [ref_action_type, True, True]

def action_2_format(step_data):
    # 把test数据集中的动作格式转换为计算matching score的格式
    action_type = step_data["action_type_id"]

    if action_type == 4:
        if step_data["action_type_text"] == 'click':  # 点击
            action_type = 'click'
            attr = {'target': step_data["touch"][::-1]} # xy -> yx aitw的标注是这种yx形式
        else:  # 上下左右滑动
            direction = step_data["action_addition"].split()[-1]
            action_type = 'swipe'
            attr = {'direction': direction}
    elif action_type == 3:
        action_type = 'input_text'
        attr = {'text': step_data['type_text']}
    elif action_type == 5:
        action_type = 'navigate_back'
        attr = None
    elif action_type == 6:
        action_type = 'navigate_home'
        attr = None
    elif action_type == 7:
        action_type = 'enter'
        attr = None
    elif action_type == 10:
        action_type = 'status'
        attr = {'goal_status': 'successful', 'answer': ''}
    elif action_type == 11:
        action_type = 'status'
        attr = {'goal_status': 'infeasible', 'answer': ''}
    else:
        typed_text = ""

    return action_type, attr


def pred_2_format_seeclick(action_pred):
    # 把模型输出的内容转换为计算action_matching的格式
    action_type = action_pred["action_type"]

    if action_type == 4:  # 点击
        action_type_new = 'click'
        attr = {'target': action_pred["click_point"][::-1]}
    elif action_type == 0: # swipe up/down/left/right are assigned the ids 1, 0, 8, and 9 respectively.
        action_type_new = 'swipe'
        attr = {'direction': 'up'}
    elif action_type == 1:
        action_type_new = 'swipe'
        attr = {'direction': 'down'}
    elif action_type == 8:
        action_type_new = 'swipe'
        attr = {'direction': 'right'}
    elif action_type == 9:
        action_type_new = 'swipe'
        attr = {'direction': 'left'}
    elif action_type == 3:
        action_type_new = 'input_text'
        attr = {'text': action_pred['typed_text'].lower()}
    elif action_type == 5:
        action_type_new = 'navigate_back'
        attr = None
    elif action_type == 6:
        action_type_new = 'navigate_home'
        attr = None
    elif action_type == 7:
        action_type_new = 'enter'
        attr = None
    elif action_type == 10:
        action_type_new = 'status'
        attr = {'goal_status': 'successful'}
    elif action_type == 11:
        action_type_new = 'status'
        attr = {'goal_status': 'infeasible'}
    else:
        raise Exception("unknown action!")

    return action_type_new, attr


def pred_2_format_autogui(action_pred, scale):
    # 把模型输出的内容转换为计算action_matching的格式
    action_type = action_pred["action_type"]

    attr = {}
    if action_type == 'click':  # 点击
        attr = {'target': list(map(lambda x: x / scale, action_pred["target"][::-1]))} # xy -> yx
    elif action_type == 'swipe': # swipe up/down/left/right are assigned the ids 1, 0, 8, and 9 respectively.
        attr = {'direction': action_pred['direction']}
    elif action_type == 'input_text':
        attr = {'text': action_pred['text'].lower()}
    elif action_type == 'status':
        attr = {'goal_status': action_pred['goal_status']}

    return action_type, attr