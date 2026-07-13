# Palette's UX Journal - Critical Learnings

## 2025-07-13 - [egui accessibility and hover tooltip pattern]
**Learning:** In egui-based interfaces (AccessKit backed), standard buttons lack native screen-reader labels unless they have tooltip/hover text or explicit labels. Chaining `.on_hover_text("...")` before `.clicked()` or `.changed()` on `ui.button(...)` or slider widgets serves as the primary way to provide screen-reader accessible descriptions and visual tooltips in a single unified immediate-mode pattern.
**Action:** Always chain `.on_hover_text("...")` directly to icon-only or short-labeled control buttons (like Play, Pause, Step, Add, Delete) to ensure visual clarity and keyboard/screen-reader accessibility. For destructive actions, combine this with `egui::RichText::new("❌").color(egui::Color32::LIGHT_RED)` to signal danger.
