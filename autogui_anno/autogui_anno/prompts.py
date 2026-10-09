"""Prompt library and functionality-text cleaning helpers for autogui_anno.

Prompt constants and the ``make_reject_prompt`` / ``make_verif_prompt`` builders
are copied verbatim from ``WebpageFunctionality/prompt_lib.py``. The text-cleaning
helpers ``get_clean_func`` / ``find_first_verb`` and the ``INVALID_MENTION`` regex
are ported from ``WebpageFunctionality/utils/tools.py``.

spacy's ``en_core_web_sm`` model is loaded lazily (via ``_get_nlp``) so that
importing this module never requires the model to be present. Install it with
``python -m spacy download en_core_web_sm`` for the verify stage's text cleaning.
"""

import re

# Func prediction
REASONING_MARK = "Reasoning"
SUMMARY_MARK = "Summary"

QUESTION = f"""Now please think step-by-step, examine the content variation in detail, and finally summarize the high-level usage of the interacted element. Do follow the output format "{REASONING_MARK}: ... {SUMMARY_MARK}: This element ..."."""

PREDICT_PROMPT_AXTREE = f"""Objective: As an Internet expert, your task is to describe the usage and functionality of a webpage element based on the changes observed in the webpage contents before and after interacting with the element.

Instructions:
1. You will be shown line-by-line differences between the webpage content before and after interacting with the element. Here's what each prefix indicates:
Unchanged: Lines that are identical before and after the interaction.
Added: New lines that appear after the interaction.
Deleted: Lines that were present before the interaction but removed afterward.
Renaming: Lines indicating elements that were renamed due to the interaction.
Attribute Update: Lines showing elements whose attributes were updated during the interaction.
Repositioned: Elements that were moved to a different part of the webpage.
2. You MUST thoroughly analyze the changes in webpage content (Added, Deleted, Unchanged lines) caused by interacting with the element, present a detailed reasoning process elucidating how the element affects the webpage, and finally summarize the element's overall purpose based on your analysis.
3. Avoid detailing every specific functionality of the webpage element. Instead, focus on its broader impact on the webpage experience. For example, if interacting with a "Products" button reveals a dropdown menu, do not catalog the subsequent webpage changes in exhaustive detail.
4. Your output MUST follow this format:
{REASONING_MARK}: (Examine the webpage variation carefully to figure out how the interacted element changes the webpage)
{SUMMARY_MARK}: This element ... (Provide a concise high-level description of the element's function. This description should contain the meaningful feature of the element in its context.)
5. (Important) Avoid mentioning redundant elements from the webpage before interaction in the Summary. For instance, rather than detailing a transition from one element to another, like "This element switches the focus from the 'Destination' input field to the 'Check-in Date' field, providing a date picker", simplify it to "This element triggers a date picker for users to select their check-in date" This approach keeps the description concise and focused on functionality, eliminating unnecessary details.
6. (Also important) To ensure uniqueness, your functionality description should reflect the instance-specific context of the element whenever possible. For example, instead of predicting 'This element adds a product to the cart,' you should predict 'This element adds a blue shirt to the cart,' where 'blue shirt' is specific to the current instance. Similarly, rather than predicting 'This element facilitates the selection of an hour for the return time,' you should predict 'This element updates the return time to 13 p.m.' if such information is directly available. Ensure that the description remains accurate, grounded in visible data, and does not speculate on unseen values."""\
+ """{exemplar}
Now it's your turn.
Element interaction: After {action_str}, the webpage exhibits the following variations:
{variation}

"""\
+QUESTION

PREDICT_PROMPT_SCHEMA = f"""Objective: As an Internet expert, your task is to describe the usage and functionality of a webpage element based on the changes observed in the webpage contents before and after interacting with the element.

Instructions:
1. You will be shown line-by-line differences between the webpage content before and after interacting with the element. Here's what each prefix indicates:
Unchanged: Lines that are identical before and after the interaction.
Added: New lines that appear after the interaction.
Deleted: Lines that were present before the interaction but removed afterward.

Following the prefix, each line comprises three components of an element: Element type, Displayed text, and Coordinates of the element's bounding box  (X, Y, Width, Height). The coordinates provide essential positional information of the webpage elements so please utilize these for reasoning.

2. You MUST thoroughly analyze the changes in webpage content (Added, Deleted, Unchanged lines) caused by interacting with the element, present a detailed reasoning process elucidating how the element affects the webpage, and finally summarize the element's overall purpose based on your analysis.
3. Avoid detailing every specific functionality of the webpage element. Instead, focus on describing its broader impact on the webpage experience. For example, if interacting with a "Products" button reveals a dropdown menu, do not catalog the subsequent webpage changes in exhaustive detail.
4. Your output MUST follow this format:
{REASONING_MARK}: (Examine the webpage variation carefully to figure out how the interacted element changes the webpage)
{SUMMARY_MARK}: (Provide a concise high-level description of the element's function)
5. Avoid mentioning specific elements from the webpage prior to interaction in the Summary. Instead, focus directly on the outcome of the interaction. For instance, rather than detailing a transition from one element to another, like "This element switches the focus from the 'Destination' input field to the 'Check-in Date' field, providing a date picker", simplify it to "This element triggers a date picker for users to select their check-in date." This approach keeps the description concise and focused on functionality, eliminating unnecessary details.""" + \
"""{exemplar}
Now it's your turn.
Element interaction: After {action_str}, the webpage exhibits the following variations:
{variation}

"""+QUESTION

PREDICT_PROMPT_AXTREE_ANDROID = f"""Objective: As an expert in using mobile devices, your task is to describe the usage and functionality of a UI element based on the changes observed in the UI contents before and after interacting with the element.

Instructions:
1. You will be shown line-by-line differences between the UI content before and after interacting with the element. Here's what each prefix indicates:
Unchanged: Lines that are identical before and after the interaction.
Added: New lines that appear after the interaction.
Deleted: Lines that were present before the interaction but removed afterward.
In addition, the interacted element will be marked in the shown differnces with "(This is the interacted element)".
2. You MUST thoroughly analyze the changes in UI content (Added, Deleted, Unchanged lines) caused by interacting with the element, present a detailed reasoning process elucidating how the element affects the UI, and finally summarize the element's overall purpose based on your analysis.
3. Avoid detailing every specific functionality of the element. Instead, focus on describing its broader impact on the UI experience. For example, if interacting with a "Products" button reveals a dropdown menu, do not catalog the subsequent UI changes in exhaustive detail.
4. Your output MUST follow this format:
{REASONING_MARK}: (Examine the UI variation carefully to figure out how the interacted element changes the UI)
{SUMMARY_MARK}: This element ... (Provide a concise high-level description of the element's function. This description should contain the meaningful feature of the element in its context.)
5. (Important) Avoid mentioning redundant elements from the UI before interaction in the Summary. For instance, rather than detailing a transition from one element to another, like "This element switches the focus from the 'Destination' input field to the 'Check-in Date' field, providing a date picker," simplify it to "This element triggers a date picker for users to select their check-in date." This approach keeps the description concise and focused on functionality, eliminating unnecessary details.
6. (Also important) To ensure uniqueness, your functionality description should reflect the instance-specific context of the element whenever possible. For example, instead of predicting 'This element adds a product to the cart,' you should predict 'This element adds a blue shirt to the cart,' where 'blue shirt' is specific to the current instance. Similarly, rather than predicting 'This element facilitates the selection of an hour for the return time,' you should predict 'This element updates the return time to 13 p.m.' if such information is directly available. Ensure that the description remains accurate, grounded in visible data, and does not speculate on unseen values."""\
+ """{exemplar}
Now it's your turn.
Element interaction: After {action_str}, the mobile phone UI exhibits the following variations:
{variation}

"""\
+QUESTION

# Prompt for describing webpages
DESCRIPTION_MARK = "Overall Functionality"

DESCRIBING_QUERY = f"""Please observe the webpage carefully, describe the webpage regions hierarchically, and finally summarize the overall webpage functionality. Do follow the output format "Region 1: ... Region 2: ... {DESCRIPTION_MARK}: ..."."""

DESCRIBING_PROMPT = f"""You are an Internet expert. You are tasked with describing the functionality of a webpage element according to the webpage contents before and after interacting with the element.

Requirements:
1. You will be provided with the accessibility trees of the webpages. You need to examine the webpage content carefully to discern the webpage hierarchy so that you can describe the functions provided by different page regions.
2. You must describe the high-level functionality of each page region and then summarize the overall functionality of the whole webpage.
The desirable output:
Region 1 (region name): (high-level regional functionality)
Region 2 (region name): (high-level regional functionality)
...
{DESCRIPTION_MARK}: (summarize the high-level functionality of the entire webpage and avoid mentioning specific elements)
3. You must describe the main content region in more detail but briefly summarize less meaningful regions, such as headers, navigation bars, and footers (if they exist), in a single sentence.
4. The final summary must emphasize the distinctive features of the webpage and not be too vague.""" + \
"""
{exemplar}
Now it's your turn.
Webpage content:
{content}

""" + DESCRIBING_QUERY


DESCRIPTION_EXEMPLAR = f"""Here is an example:
Webpage content:
[1] RootWebArea 'Postmill' focused: True
 [2] HeaderAsNonLandmark ''
  [3] link 'Home'
 [5] link 'Forums'
 [7] link 'Wiki'
 [9] searchbox 'Search query'
 [10] link 'Notifications (0)'
 [11] link 'Submit'
 [13] button 'MarvelsGrantMan136' hasPopup: menu expanded: False
 [15] main ''
  [16] link 'Submissions'
  [18] link 'Comments'
  [20] button 'Filter on: Subscribed' hasPopup: menu expanded: False
  [22] button 'Sort by: Hot' hasPopup: menu expanded: False
  [24] article ''
   [25] HeaderAsNonLandmark ''
    [26] heading 'Virtual Book Reading Meetup on March 15th'
     [27] link 'Virtual Book Reading Meetup on March 15th'
    [29] StaticText 'Submitted by '
    [30] link 'MarvelsGrantMan136' expanded: False
    [33] time 'February 27, 2024 at 7:53:20 PM EST'
     [34] StaticText '3 hours ago'
    [35] StaticText ' in '
    [36] link 'funny'
   [38] link 'No comments'
   [40] link 'Edit'
   [42] button 'Delete'
   [44] button 'Retract upvote'
   [45] StaticText '1'
   [46] button 'Downvote'

{DESCRIBING_QUERY}

Description:
Region 1 (Header Section): The top section of the page includes essential navigation links such as Home, Forums, and Wiki, alongside a search function for content discovery. User-specific features like notifications and content submission options suggest a personalized experience. Additionally, there's a user account button indicating more personalized options.

Region 2 (Main Content Area): The core of the webpage focuses on user-generated content, with features for filtering and sorting submissions to tailor the content display according to user preferences. It highlights user contributions, like a post about a virtual book reading meetup, providing context such as the author, submission time, and associated forum. Users can interact with posts through comments, upvotes, and downvotes, as well as edit or delete their submissions.

Overall Webpage Functionality:
The webpage serves as a community platform where users can share, discover, and discuss content of mutual interest. The layout promotes ease of navigation through organized links and a structured main content area that highlights user submissions and facilitates interaction through comments and votes. The header ensures users can quickly access various site sections and manage their interactions and contributions. The main area emphasizes community-driven content, with tools to filter, sort, and directly engage with that content, supporting an interactive and personalized user experience."""

# Predicting with descriptions
PREDICT_WITH_DESCS = f"""Objective: Your mission, as a digital navigation specialist, is to deduce and articulate the function and usage of a specific webpage element. This deduction should be based on your analysis of the differences in webpage content before and after interacting with said element.

Instructions:
1. You will be given descriptions of a webpage before and after interaction with an element. Your primary task is to meticulously analyze the differences in content resulting from this interaction to understand what the functionality of the element is in the webpage context.
2. You must present a detailed reasoning process before finally summarizing the element's overall purpose based on your analysis.
3. Prioritize examining changes in the webpage's regional content over individual element variations. This approach will provide a more holistic view of the element's impact on the webpage.
4. You should emphasize on the main content changes and pay less attention to less meaningful regions, such as headers, navigation bars, and footers.
5. Your output MUST follow this format:
Reasoning: (Examine the webpage variation carefully to figure out how the interacted element changes the webpage)
{SUMMARY_MARK}: This element ... (Provide a high-level description of the element's functionality. This description should contain the meaningful feature of the element in its context.)
6. Avoid mentioning specific elements from the webpage prior to interaction in the Summary. Instead, focus directly on the outcome of the interaction. For instance, rather than detailing element changes, like "This element triggers the disappearance of the header and language selector elements and the emergence of a login form", simplify it to "This element triggers the display of a login page." This approach keeps the summary concise and focused on functionality, eliminating unnecessary details.
7. The final summary must emphasize the distinctive functionality of the element and not be too vague. For example, "This element redirects the user to a specific website, allowing them to access its content and services"  is too broad and lacks meaningful detail. Instead, it should precisely describe how the element provides users with unique value""" + \
"""{exemplar}

Now it's your turn.
The description of the initial webpage:
{before}

After {action_str}, we jump to a new webpage.
The description of the new webpage is:
{after}

""" +  QUESTION

REJECT_PROMPT_V4 = """Your primary objective is to determine whether the changes in the webpage's content are sufficient for predicting the functionality of the webpage element causing these changes after being interacted with.

Instructions:
1. You will be shown the outcome (webpage changes) resulting from interacting with the element. The outcome can take one of two forms:  changes to the webpage description, or line-by-line differences. For the latter form, here's what each prefix indicates:
Unchanged: Lines that are identical before and after the interaction.
Added: New lines that appear after the interaction.
Deleted: Lines that were present before the interaction but removed afterward.
Renaming: Lines indicating elements that were renamed due to the interaction.
Attribute Update: Lines showing elements whose attributes were updated during the interaction.
Repositioned: Elements that were moved to a different part of the webpage.
2. Analyze the provided outcome and provide detailed reasoning for whether this outcome helps to predict the element's functionality, considering the following stringent criteria:
1) Explicitness of Changes: Rate how directly the changes suggest the element's functionality. Score 1-{MAX_SCORE} for clear, unambiguous changes. Clearer changes obtain a higher score. No scores for vague, meaningless, or non-specific changes.
Positive Example: A button labeled "Show More" that, upon interaction, clearly adds new content below it. The direct addition of content clearly indicates a content expansion functionality. Score: {MAX_SCORE}
Negative Example: After clicking a "Details" button, the page layout changes subtly without adding relevant information or altering content in a meaningful way. The changes do not clearly relate to the button's presumed functionality. Score: 0

2) Relevance of Changes: Evaluate the significance of the modifications in relation to the element's intended function. Score 1-{MAX_SCORE} for changes that enhance understanding of the element's role. Highly related modifications obtain a high score. No scores for irrelevant or unrelated content changes.
Positive Example: Clicking on a "Contact Us" button opens a form to fill out, which is highly relevant to the button's intended functionality. Score: {MAX_SCORE}
Negative Example: Clicking on a "View Profile" link leads to a page refresh without displaying the profile or any related information, making the change irrelevant to the link's intended purpose. Score: 0

3) Predictability of Outcome: Assess how anticipated the interaction outcome is based on the changes, considering common web conventions and user experience principles. Score 1-{MAX_SCORE} for highly predictable outcomes. Highly predictable changes obtain a high score. No scores for outcomes that are moderate, unexpected, or counterintuitive.
Positive Example: Clicking or hovering over a "Help" icon reveals a tooltip with information, an outcome that users can easily predict given the icon's universal symbol for help. Score: {MAX_SCORE}
Negative Example: Hovering over a "Settings" icon changes its color but does not display any settings options, tooltips, or lead to a settings page, making the outcome unpredictable and the icon's purpose unclear. Score: 0

Provide a detailed analysis based on these criteria, ending with an overall score between 0-{TOTAL_MAX_SCORE}.

4. Your output MUST follow this format:
Reasoning:
- Explicitness of Changes: [Detailed reasoning and score]
- Relevance of Changes: [Detailed reasoning and score]
- Predictability of Outcome: [Detailed reasoning and score]

Overall score: <score>[criterium 1 score] + [criterium 2 score] + [criterium 3 score] = [total score]</score> (output an additive equation enclosed with a pair of <score></score>)

Now it's your turn.
The interacted element is {target_element}
Outcome:
{outcome}

Now please think step-by-step, examine the content variation in detail, and finally calculate scores according to the criteria. Do follow the output format "Reasoning: ... Overall score: <score>...</score>"."""

def make_reject_prompt(max_score):
    return REJECT_PROMPT_V4.replace("{MAX_SCORE}", str(max_score)).replace("{TOTAL_MAX_SCORE}", str(3*max_score))

REJECT_PROMPT_ANDROID = """Your primary objective is to determine whether the changes in the Android UI's content are sufficient for predicting the functionality of the UI element causing these changes after being interacted with.

Instructions:
1. You will be shown the outcome (UI changes in the format of line-by-line differences) resulting from interacting with the element. Here's what each prefix indicates:
Unchanged: Lines that are identical before and after the interaction.
Added: New lines that appear after the interaction.
Deleted: Lines that were present before the interaction but removed afterward.
2. Analyze the provided outcome and provide detailed reasoning for whether this outcome helps to predict the element's functionality, considering the following stringent criteria:
1) Explicitness of Changes: Rate how directly the changes suggest the element's functionality. Score 1-3 for clear, unambiguous changes. Clearer changes obtain a higher score. No scores for vague, meaningless, or non-specific changes.
Positive Example: A button labeled "Show More" that, upon interaction, clearly adds new content below it. The direct addition of content clearly indicates a content expansion functionality. Score: 3
Negative Example: After clicking a "Details" button, the page layout changes subtly without adding relevant information or altering content in a meaningful way. The changes do not clearly relate to the button's presumed functionality. Score: 0

2) Relevance of Changes: Evaluate the significance of the modifications in relation to the element's intended function. Score 1-3 for changes that enhance understanding of the element’s role. Highly related modifications obtain a high score. No scores for irrelevant or unrelated content changes.
Positive Example: Clicking on a "Contact Us" button opens a form to fill out, which is highly relevant to the button's intended functionality. Score: 3
Negative Example: Clicking on a "View Profile" link leads to a page refresh without displaying the profile or any related information, making the change irrelevant to the link's intended purpose. Score: 0

3) Predictability of Outcome: Assess how anticipated the interaction outcome is based on the changes, considering common web conventions and user experience principles. Score 1-3 for highly predictable outcomes. Highly predictable changes obtain a high score. No scores for outcomes that are moderate, unexpected, or counterintuitive.
Positive Example: Clicking a "Help" icon reveals a tooltip with information, an outcome that users can easily predict given the icon's universal symbol for help. Score: 3
Negative Example: Clicking a "Settings" icon changes its color but does not display any settings options, tooltips, or lead to a settings page, making the outcome unpredictable and the icon's purpose unclear. Score: 0

Provide a detailed analysis based on these criteria, ending with an overall score between 0-9.

4. Your output MUST follow this format:
Reasoning:
- Explicitness of Changes: [Detailed reasoning and score]
- Relevance of Changes: [Detailed reasoning and score]
- Predictability of Outcome: [Detailed reasoning and score]

Overall score: <score>[criterium 1 score] + [criterium 2 score] + [criterium 3 score] = [total score]</score> (output an additive equation enclosed with a pair of <score></score>)

Now it's your turn.
The interacted element is {target_element}
Outcome:
{outcome}

Now please think step-by-step, examine the content variation in detail, and finally calculate scores according to the criteria. Do follow the output format "Reasoning: ... Overall score: <score>...</score>"."""

# Cycle-consistency checking
CYCLE_CONSISTENCY_PROMPT = """Given the following inputs:

1) Webpage Accessibility Tree: This input represents the hierarchical structure of a webpage's elements, emphasizing semantic information and relationships important for accessibility. Each node in the tree includes details such as the element's role (e.g., button, link, heading), relevant attributes (e.g., expanded), and hierarchical relationships with other elements.

2) Task Description: This input specifies the functionality of the element that needs to be identified within the accessibility tree. The description will outline the action that a user aims to perform (e.g., submit a form, navigate to a specific section, adjust settings) or the information they seek (e.g., a specific piece of content, form field).

Your job is to:
1) Analyze the provided accessibility tree to understand the structure and semantics of the webpage's elements.
2) Based on the task description, identify the element within the accessibility tree that corresponds to the specified functionality. Consider the role, attributes, and hierarchical position of elements to determine the most relevant match.
3) Finally return the chosen element by outputting its integer label enclosed with a pair of <labels></labels>. If you think multiple elements correspond to the given functionality, just return a Python list containing the labels, e.g., <labels>[..., ...]</labels>. If you are unable to determine with the given information, please return an empty list, e.g., <labels>[]</labels>.
4) You Must think step-by-step before presenting your answer. Do follow this output format:
Reasoning: (reason about the accessibility tree and the target element whose functionality is specified in the given task)
Seleted element: (the label of the target element required by the task)

Example:
Webpage Accessibility Tree:
RootWebArea "mens black hoodie | Kohl's" focused: True
[1] button 'Category' expanded: False
[2] button 'Brand' expanded: False
[3] button 'Size Range' expanded: False
[4] button 'Material' expanded: False

Task Description: Please identify the target element. The element serves as a filter for users to narrow down their search results based on different size categories.

Reference response:
Reasoning: Analyzing the provided accessibility tree, we focus on identifying the element that enables filtering by size categories. The tree lists buttons with labels indicative of their filtering functionality: 'Category', 'Brand', 'Size Range', and 'Material'. Among these, the 'Size Range' button is directly aligned with the task's need to filter search results based on size, making it the clear choice for the target element.
Selected element: <labels>[3]</labels>

Now it's your turn.
Webpage Accessibility Tree:
{content}

Task Description: Please identify the target element. This element {functionality}

Your answer:"""

# Cycle-consistency checking
criteria_list = [
   "The element does not support the action in any capacity.",
   "The element hardly provides support for the action.",
"The element provides minimal support for the action.",
"The element supports the action but with limitations.",
"The element fully supports the action without significant limitations."
]
CYCLE_CONSISTENCY_PROMPT_V2 = """Given the following inputs:
1) Webpage content: This input represents the hierarchical structure of a webpage's elements, emphasizing semantic information and relationships. Each node in the tree includes details such as the element's role (e.g., button, link, heading), relevant attributes (e.g., expanded), and hierarchical relationships with other elements.

2) Task Description: This describes the action a user intends to perform (such as submitting a form, navigating to a particular section, or adjusting settings) or the information they seek (such as a specific content piece or form field). It also introduces a candidate element for evaluation and then presents the webpage changes caused by interacting with this element. Your task is to assess whether this element effectively facilitates the specified user action.

Your job is to:
1) Analyze the provided webpage content to understand the structure and semantics of the webpage's elements.
2) Evaluate the Candidate Element: Determine the suitability of the specified candidate element for the described action. Consider the element's role, attributes, and position within the hierarchy. Your evaluation should be grounded in how well these aspects align with the required functionality for the user's intended action.
3) Score the Element: Assign a score ranging from 0 to {max_score}, enclosed within <score></score> tags. This score should reflect the degree to which the candidate element meets the action's requirements:
{criteria}
4) Provide Reasoning: Before presenting your score, offer a detailed explanation of your reasoning. This should cover your analysis of the webpage content, the relationship between the candidate element and the specified action, and how these factors informed your scoring decision.
5) Format for Your Answer:
Reasoning: [Provide a comprehensive analysis covering the webpage's insights, the relationship between the specified action and the candidate element, and the rationale behind your scoring decision.]

Score: <score>[Your Score Here]</score>

Example:
Webpage content:
[0] RootWebArea 'Rental Cars at Low, Affordable Rates'
[1] dialog 'Vehicle Class' modal: True
[2] radiogroup 'Cars' 
[3] 	radio 'Economy Car' checked: false
[4] 	radio 'Compact Car' checked: false
[5] 	radio 'Full Size SUV' checked: false
[6] 	radio 'Premium & Luxury SUV' checked: false
[7] 	radio 'Jeeps' checked: false
[8] radiogroup 'Vans' 
[9] 	radio 'Minivans' checked: false
[10] 	radio 'Passenger Vans' checked: false
[11] 	radio 'Cargo Vans' checked: false
[12] 	radio 'Commercial & Cargo Vans' checked: false
[13] radiogroup 'Trucks' 
[14] 	radio 'Small Pickup' checked: false
[15] 	radio 'Full Size Pickup' checked: false
[16] button 'Cancel'
[17] button 'Apply Filter'

Task Description: Please identify the target element. The element helps users narrow down their vehicle choices to minivans specifically

Candidate element: [9] 	radio 'Minivans'
After interacting with the candidate element, the webpage exhibits these changes:
Upon clicking the "Minivans" <input> element, a new "Remove Filter" button is added to the dialog modal. The radio buttons for different vehicle classes remain unchanged, but the "Minivans" radio button is checked after the interaction.

Reference response:
Reasoning: The provided webpage content outlines a clear hierarchical structure for selecting vehicle types on a car rental webpage, categorized into sections like Cars, Vans, and Trucks, each with its own set of options represented as radio buttons. 
The task involves narrowing vehicle choices to minivans. The candidate element, a radio button labeled 'Minivans', is part of the 'Vans' radiogroup on a car rental webpage. This directly supports the user's action of narrowing choices to minivans, aligning with the task description. The element's interaction leads to its checking and introduces a "Remove Filter" button, offering users flexibility. Its role, position, and functionality fully support the user's intent without any identified limitations.

Score:
<score>{max_score}</score>

Now it's your turn.
Current webpage content:
{content}

Task Description: Please identify the target element. This element {functionality}

Candidate element: {candidate}
{outcome_info}

Now please do your job. Your answer:"""

def make_verif_prompt(max_score):
   criteria = criteria_list[:1] + criteria_list[-max_score:]
   return CYCLE_CONSISTENCY_PROMPT_V2.replace('{max_score}', str(max_score)).replace('{criteria}', '\n'.join(f'{idx}: {criterium}' for idx, criterium in enumerate(criteria)))

CYCLE_CONSISTENCY_PROMPT_ANDROID = """Given the following inputs:
1) Android UI content: This input represents the hierarchical structure of a UI's elements, emphasizing semantic information and relationships. Each node in the tree includes details such as the element's role, relevant attributes (e.g., clickable), and hierarchical relationships with other elements.

2) Task Description: This describes the action a user intends to perform (such as submitting a form, navigating to a particular section, or adjusting settings) or the information they seek (such as a specific content piece or form field). It also introduces a candidate element for evaluation and then presents the UI changes caused by interacting with this element. Your task is to assess whether this element effectively facilitates the specified user action.

Your job is to:
1) Analyze the provided UI content to understand the structure and semantics of the UI's elements.
2) Evaluate the Candidate Element: Determine the suitability of the specified candidate element for the described action. Consider the element's role, attributes, and position within the hierarchy. Your evaluation should be grounded in how well these aspects align with the required functionality for the user's intended action.
3) Score the Element: Assign a score ranging from 0 to 3, enclosed within <score></score> tags. This score should reflect the degree to which the candidate element meets the action's requirements:
0: The element does not support the action in any capacity.
1: The element provides minimal support for the action.
2: The element supports the action but with limitations.
3: The element fully supports the action without significant limitations.
4) Provide Reasoning: Before presenting your score, offer a detailed explanation of your reasoning. This should cover your analysis of the UI content, the relationship between the candidate element and the specified action, and how these factors informed your scoring decision.
5) Format for Your Answer:
Reasoning: [Provide a comprehensive analysis covering the UI's insights, the relationship between the specified action and the candidate element, and the rationale behind your scoring decision.]

Score: <score>[Your Score Here]</score>

Example:
UI content:
[0] ViewGroup text: '' resource-id: com.android.settings:id/settings_homepage_container
[1] 	LinearLayout text: '' resource-id: com.android.settings:id/homepage_app_bar_two_pane_view
[2] 		TextView text: 'Search' resource-id: com.android.settings:id/search_bar_title
[3] 		ImageView text: '' resource-id: com.android.settings:id/account_avatar_two_pane_version, clickable: true
[4] 	RecyclerView text: '' resource-id: com.android.settings:id/recycler_view
[5] 		LinearLayout text: '' clickable: true
[6]  			TextView text: 'Network & internet' resource-id: android:id/title
[7]  			TextView text: 'Mobile, WiFi, hotspot' resource-id: android:id/summary
[8] 		LinearLayout text: '' clickable: true (This is the interacted element)
[9] 			TextView text: 'Connected devices' resource-id: android:id/title
[10] 			TextView text: 'Bluetooth, pairing' resource-id: android:id/summary
[11] LinearLayout text: '' clickable: true
[12] 	TextView text: 'Apps' resource-id: android:id/title
[13] 	TextView text: 'Assistant, recent apps, default apps' resource-id: android:id/summary
[14] LinearLayout text: '' clickable: true
[15] 	TextView text: 'Notifications' resource-id: android:id/title
[16] 	TextView text: 'Notification history, conversations' resource-id: android:id/summary
[17] LinearLayout text: '' clickable: true
[18] 	TextView text: 'Battery' resource-id: android:id/title
[19] 	TextView text: '100%' resource-id: android:id/summary

Task Description: Please identify the target element. This element navigates to a Bluetooth settings screen, allowing users to manage their Bluetooth connections and devices.

Candidate element: [8] LinearLayout text: '' clickable: true
After interacting with the candidate element, the UI exhibits these changes:
The UI content primarily shifts from a general settings overview to a more focused view. The deletion of several high-level categories like "Network & internet," "Connected devices," "Apps," and "Battery" indicates a transition away from the main settings page. Instead, new content is added, including options such as "Pair new device" and "Saved devices" along with details like visibility to other devices.

Reference response:
Reasoning: The UI content reveals a hierarchical structure where the candidate element, located at position [8], is a clickable LinearLayout containing text elements that indicate its purpose. The text within this element—'Connected devices' and 'Bluetooth, pairing'—clearly suggests its relevance to Bluetooth settings, aligning well with the user's task.

Upon interacting with this element, the UI transitions from a general settings overview to a more specific Bluetooth-related view. The appearance of options such as 'Pair new device' and 'Saved devices,' along with details about device visibility, confirms that the interaction successfully leads the user to a Bluetooth settings screen. This supports the intended action of managing Bluetooth connections and devices.

Given the accurate alignment between the element's content, its clickable nature, and the resulting UI change, the candidate element effectively facilitates the specified user action without any notable limitations.

Score: <score>3</score>

Now it's your turn.
Current UI content:
{content}

Task Description: Please identify the target element. This element {functionality}

Candidate element: {candidate}
{outcome_info}

Now please do your job. Your answer:"""


# ---------------------------------------------------------------------------
# Functionality-text cleaning helpers (ported from WebpageFunctionality/utils/tools.py)
#
# spacy's en_core_web_sm model is loaded lazily so that importing this module
# never requires the model to be present. Install the model for the verify
# stage's text cleaning with: python -m spacy download en_core_web_sm
# ---------------------------------------------------------------------------

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def find_first_verb(text):
    # Process the text
    doc = _get_nlp()(text)
    for token in doc:
        if token.pos_ in ["VERB", 'AUX', "MD"] and token.tag_ not in ["VBD", "VBN"] and token.dep_ not in ["amod", "acomp"]:  # Check if the token is a verb, auxiliary verb such as "is" or "are," or modal verb. The verb should not be used as an adjective.
            # Find the start position of the verb in the paragraph
            start_index = token.idx
            return token.text, start_index
    return "No verb found", -1  # Return a message and -1 if no verb is found

INVALID_MENTION = re.compile(r'"[^"]*"\s*|\'[^\']*\'\s*|<[^>]*>\s*')
def get_clean_func(elem_func: str):
    double_linebreak_id = elem_func.find('\n\n')
    func_start = elem_func.find(f"{SUMMARY_MARK}:", double_linebreak_id)
    if func_start == -1:
        func_start = elem_func.find("Functionality:", double_linebreak_id)

    summary = elem_func[elem_func.find(':', func_start)+1:].strip()
    func_end = summary.find('\n\n', func_start)
    if func_end == -1: func_end = 9999

    elem_func_summary = summary[:func_end]

    # Remove any mentions of the element's displayed text and its role to avoid cheating
    elem_func_summary = INVALID_MENTION.sub('', elem_func_summary[:60]) + elem_func_summary[60:]

    # Exrtract the element func. starting from the first verb
    _, index = find_first_verb(elem_func_summary)

    func = elem_func_summary[index:func_end]

    return func
