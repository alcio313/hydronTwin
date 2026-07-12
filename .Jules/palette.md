# Palette's Journal - Critical UX/Accessibility Learnings

## 2025-07-12 - [Unified Tooltip and Accessible Descriptions in egui]
**Learning:** In egui-based Rust applications, adding `.on_hover_text(...)` to buttons and other interactive elements is the standard and most robust way to provide accessible tooltips, descriptions, and descriptive hover labels for both mouse-hover users and screen readers.
**Action:** Always append `.on_hover_text(...)` to critical simulation controls and icon-only buttons (like delete '❌') to guide users and improve screen-reader accessibility.
