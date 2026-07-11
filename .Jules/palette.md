# Palette's Journal - Critical UX/Accessibility Learnings

## 2025-02-14 - [egui Icon-Only Button Accessibility]
**Learning:** In egui immediate-mode GUI frameworks, screen readers and standard accessibility paths can lack full description for icon-only buttons (like "❌" or "↺"). Adding clear, concise `.on_hover_text(...)` or descriptive helper text is essential for screen reader support and tooltip-based clarity.
**Action:** Always append `.on_hover_text(...)` with active verb descriptions to icon-only buttons and core action buttons to guarantee accessibility and visual tooltips.
