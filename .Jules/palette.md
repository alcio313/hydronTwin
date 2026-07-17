# Palette's UX and Accessibility Journal

This journal tracks critical UX and accessibility learnings discovered during development of the HydRON app.

## 2025-02-15 - [Unified Pattern for Destructive Actions in egui]
**Learning:** Destructive/deletion buttons in this immediate-mode GUI framework (`egui`) lack default distinctive visual styling or confirmation states, leading to potential accidental clicks. Furthermore, letting users delete the last item (such as the last ground station) can violate domain safety constraints.
**Action:** Standardize destructive/delete buttons to style their text/icon using `egui::RichText::new("❌").color(egui::Color32::LIGHT_RED)`, pair them with explicit `.on_hover_text(...)` descriptions for visual tooltips and screen reader support, and wrap them in `ui.add_enabled_ui(enabled, |ui| { ... })` to enforce minimum boundary safety constraints (e.g., `self.ground_stations.len() > 1`).
