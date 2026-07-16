# Palette's Journal - Critical UX/accessibility learnings

## 2025-02-14 - Ground Station Deletion Affordance
**Learning:** Destructive actions such as deleting ground stations in simulation systems require explicit guardrails. Disabling the action when only a single node remains prevents empty-state calculation panics or completely non-functional views. Pair disabled states with helpful tooltips/hover text to explain why the action is disabled.
**Action:** Wrap deletion buttons in an `egui::Ui::add_enabled_ui` block to programmatically disable them under constraints. Style with `egui::Color32::LIGHT_RED` and append `.on_hover_text(...)` for screen-reader and visual context.
