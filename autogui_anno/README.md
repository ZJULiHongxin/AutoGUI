# autogui_anno

The AutoGUI UI-element functionality annotation pipeline (ACL 2025).

This package turns raw GUI observations into verified, task-grounded
functionality annotations through five stages. Each stage maps to a section of
the paper:

1. **collect** — gather raw GUI screenshots and accessibility trees (Data Collection).
2. **reject** — filter out low-quality or redundant elements before annotation (Element Rejection / Sampling).
3. **annotate** — generate candidate functionality descriptions for UI elements with an LLM (Functionality Annotation).
4. **verify** — validate candidate annotations for correctness and grounding (Annotation Verification).
5. **generate-tasks** — compose verified annotations into instruction-following tasks (Task Generation).

> This README is a stub; it is expanded in a later task.
