"""PySide6 desktop editor with live preview for feature diagrams."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStyle,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QCheckBox,
)

from feature_diagram_core.models import ModelParseError

from .dialogs import ask_feature, ask_relation
from .preview import DiagramGraphicsView
from .state import DiagramDocument, FeatureItem, RelationItem, relation_to_display
from .widgets import DropdownOption, ListRowWidget, ReorderListWidget, SearchableComboBox


APP_DARK_STYLESHEET = """
QWidget {
  background: #101722;
  color: #e6edf8;
  font-family: "SF Pro Text", "Inter", "Segoe UI", sans-serif;
  font-size: 13px;
}
QLabel, QCheckBox {
  background: transparent;
}
QMainWindow {
  background: #0d1420;
}
QToolBar {
  background: #121c2b;
  border: 1px solid #1f2a3f;
  border-radius: 10px;
  spacing: 4px;
  padding: 3px;
}
QToolButton, QPushButton {
  background: #172233;
  border: 1px solid #273651;
  border-radius: 7px;
  min-height: 20px;
  padding: 2px 6px;
}
QToolButton:hover, QPushButton:hover {
  background: #1f2c43;
  border-color: #395179;
}
QToolButton:pressed, QPushButton:pressed {
  background: #152033;
}
QToolButton:checked {
  background: #2b3c5a;
  border-color: #5c7dae;
}
QToolButton:disabled, QPushButton:disabled {
  color: #7088ac;
  border-color: #233148;
}
QGroupBox {
  border: 1px solid #22324b;
  border-radius: 10px;
  margin-top: 12px;
  padding: 12px 8px 8px 8px;
  font-weight: 600;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 6px;
  color: #bfd1ee;
}
QFrame#DetailPanel {
  border: 1px solid #22324b;
  border-radius: 10px;
  background: #131d2d;
}
QLineEdit, QComboBox {
  background: #1a2638;
  border: 1px solid #334765;
  border-radius: 8px;
  padding: 6px 8px;
  selection-background-color: #3b5f90;
}
QLineEdit:focus, QComboBox:focus {
  border-color: #6e94c7;
}
QListWidget {
  background: transparent;
  border: none;
  outline: none;
}
QListWidget::item {
  margin: 0px;
}
QListWidget::item:selected {
  background: #23344d;
}
QFrame#ListRow {
  background: transparent;
  border: none;
  border-bottom: 1px solid #27374f;
}
QFrame#ListRow[reference="true"] {
  border: 1px dashed #86a8d6;
  border-radius: 4px;
}
QLabel#RowHandle {
  color: #9cb3d8;
  font-size: 13px;
}
QLabel#RowTitle {
  font-size: 13px;
  font-weight: 600;
  color: #ecf3ff;
}
QLabel#RowSubtitle {
  font-size: 12px;
  color: #a1b7d7;
}
QLabel#RowMeta {
  font-size: 11px;
  color: #859cc0;
}
QToolButton#RowDelete {
  background: transparent;
  border: none;
  padding: 0px;
  min-height: 18px;
}
QToolButton#RowDelete:hover {
  background: #2a3a56;
  border-radius: 5px;
}
QFrame#SearchPopup {
  background: #121d2d;
  border: 1px solid #3a5174;
  border-radius: 10px;
}
QLabel#PopupHint {
  color: #8ba3c9;
  font-size: 11px;
}
QListWidget#SearchResults {
  background: #101a29;
  border: 1px solid #2a3a57;
  border-radius: 8px;
}
QListWidget#SearchResults::item {
  border-radius: 6px;
}
QListWidget#SearchResults::item:selected {
  background: #263952;
}
QLabel#DialogTitle {
  font-size: 17px;
  font-weight: 700;
  color: #eef4ff;
}
QLabel#DialogSubtitle {
  color: #95aacb;
  font-size: 12px;
}
QDialog {
  background: #101722;
}
"""


class FeatureDiagramEditorWindow(QMainWindow):
    """Main desktop editor window."""

    def __init__(self) -> None:
        super().__init__()

        self.document = DiagramDocument()
        self.feature_edit_mode = False
        self.relation_edit_mode = False
        self.active_detail: Optional[Tuple[str, int]] = None

        self._history: List[Tuple[List[FeatureItem], List[RelationItem]]] = []
        self._history_index = -1
        self._clean_history_index = -1
        self._initial_split_applied = False

        self.setWindowTitle("Feature Diagram Editor")
        self.resize(1500, 920)
        self.setMinimumSize(1140, 740)

        self._build_ui()
        self._build_toolbar()

        self.statusBar().showMessage("Preview: scroll to zoom, drag with left mouse button to pan")

        self._reset_history()
        self._refresh_all(close_detail=True)

    def _icon(self, theme_names: List[str], fallback: QStyle.StandardPixmap) -> QIcon:
        for theme_name in theme_names:
            icon = QIcon.fromTheme(theme_name)
            if not icon.isNull():
                return icon
        return self.style().standardIcon(fallback)

    def _make_small_button(
        self,
        icon: QIcon,
        tooltip: str,
        checkable: bool = False,
        width: int = 30,
        height: int = 20,
    ) -> QToolButton:
        button = QToolButton(self)
        button.setIcon(icon)
        button.setIconSize(QSize(13, 13))
        button.setFixedSize(width, height)
        button.setToolTip(tooltip)
        button.setCheckable(checkable)
        return button

    def _snapshot_history_state(self) -> Tuple[List[FeatureItem], List[RelationItem]]:
        return deepcopy(self.document.features), deepcopy(self.document.relations)

    def _restore_history_state(self, state: Tuple[List[FeatureItem], List[RelationItem]]) -> None:
        features, relations = state
        self.document.features = deepcopy(features)
        self.document.relations = deepcopy(relations)

    def _update_dirty_from_history(self) -> None:
        self.document.set_dirty(self._history_index != self._clean_history_index)

    def _reset_history(self) -> None:
        self._history = [self._snapshot_history_state()]
        self._history_index = 0
        self._clean_history_index = 0
        self._update_dirty_from_history()

    def _push_history_state(self) -> None:
        current = self._snapshot_history_state()
        if self._history and current == self._history[self._history_index]:
            self._update_dirty_from_history()
            return

        if self._history_index < len(self._history) - 1:
            self._history = self._history[: self._history_index + 1]
            if self._clean_history_index > self._history_index:
                self._clean_history_index = -1

        self._history.append(current)
        self._history_index = len(self._history) - 1
        self._update_dirty_from_history()

    def _mark_clean_history(self) -> None:
        self._clean_history_index = self._history_index
        self._update_dirty_from_history()

    def _undo(self) -> None:
        if self._history_index <= 0:
            return
        self._history_index -= 1
        self._restore_history_state(self._history[self._history_index])
        self._update_dirty_from_history()
        self._refresh_all(close_detail=True)

    def _redo(self) -> None:
        if self._history_index >= len(self._history) - 1:
            return
        self._history_index += 1
        self._restore_history_state(self._history[self._history_index])
        self._update_dirty_from_history()
        self._refresh_all(close_detail=True)

    def _commit_mutation(self, close_detail: bool = True) -> None:
        self._push_history_state()
        self._refresh_all(close_detail=close_detail)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)

        split = QSplitter(Qt.Orientation.Horizontal, root)
        root_layout.addWidget(split)
        self.main_split = split

        editor_shell = QWidget(split)
        editor_layout = QVBoxLayout(editor_shell)
        editor_layout.setContentsMargins(0, 0, 0, 0)

        self.editor_split = QSplitter(Qt.Orientation.Vertical, editor_shell)
        editor_layout.addWidget(self.editor_split)

        lists_widget = QWidget(self.editor_split)
        lists_layout = QVBoxLayout(lists_widget)
        lists_layout.setContentsMargins(0, 0, 0, 0)
        lists_layout.setSpacing(8)

        feature_group = self._build_feature_section(lists_widget)
        relation_group = self._build_relation_section(lists_widget)

        lists_layout.addWidget(feature_group, 1)
        lists_layout.addWidget(relation_group, 1)

        self.detail_panel = self._build_detail_panel(self.editor_split)

        self.editor_split.addWidget(lists_widget)
        self.editor_split.addWidget(self.detail_panel)
        self.editor_split.setStretchFactor(0, 5)
        self.editor_split.setStretchFactor(1, 2)
        self.detail_panel.hide()

        preview_shell = QWidget(split)
        preview_layout = QVBoxLayout(preview_shell)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview = DiagramGraphicsView(preview_shell)
        preview_layout.addWidget(self.preview)

        split.addWidget(editor_shell)
        split.addWidget(preview_shell)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 2)
        split.setSizes([500, 1000])

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("File", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        self.action_load = QAction(
            self._icon(["folder-open-symbolic", "document-open"], QStyle.StandardPixmap.SP_DialogOpenButton),
            "Load",
            self,
        )
        self.action_load.triggered.connect(self._load_json)
        toolbar.addAction(self.action_load)

        self.action_save = QAction(
            self._icon(["document-save-symbolic", "document-save"], QStyle.StandardPixmap.SP_DialogSaveButton),
            "Save",
            self,
        )
        self.action_save.triggered.connect(self._save_json)
        toolbar.addAction(self.action_save)

        self.action_save_as = QAction(
            self._icon(["document-save-as", "document-save"], QStyle.StandardPixmap.SP_DriveFDIcon),
            "Save As",
            self,
        )
        self.action_save_as.triggered.connect(self._save_json_as)
        toolbar.addAction(self.action_save_as)

        toolbar.addSeparator()

        self.action_undo = QAction(
            self._icon(["edit-undo", "undo"], QStyle.StandardPixmap.SP_ArrowBack),
            "Undo",
            self,
        )
        self.action_undo.triggered.connect(self._undo)
        toolbar.addAction(self.action_undo)

        self.action_redo = QAction(
            self._icon(["edit-redo", "redo"], QStyle.StandardPixmap.SP_ArrowForward),
            "Redo",
            self,
        )
        self.action_redo.triggered.connect(self._redo)
        toolbar.addAction(self.action_redo)

        toolbar.addSeparator()

        self.action_export = QAction(
            self._icon(["document-export", "image-x-generic"], QStyle.StandardPixmap.SP_ArrowDown),
            "Export",
            self,
        )
        self.action_export.triggered.connect(self._export_svg)
        toolbar.addAction(self.action_export)

        self.action_reexport = QAction(
            self._icon(["view-refresh", "document-revert"], QStyle.StandardPixmap.SP_BrowserReload),
            "Re-export",
            self,
        )
        self.action_reexport.triggered.connect(self._reexport_svg)
        toolbar.addAction(self.action_reexport)

    def _build_feature_section(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Editor Pane - Features", parent)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.addStretch(1)

        add_btn = self._make_small_button(
            self._icon(["list-add", "plus"], QStyle.StandardPixmap.SP_FileDialogNewFolder),
            "Add feature",
        )
        add_btn.clicked.connect(self._add_feature)
        header.addWidget(add_btn)

        remove_btn = self._make_small_button(
            self._icon(["list-remove", "edit-delete"], QStyle.StandardPixmap.SP_TrashIcon),
            "Remove selected feature",
        )
        remove_btn.clicked.connect(self._remove_selected_feature)
        header.addWidget(remove_btn)

        self.feature_edit_btn = self._make_small_button(
            self._icon(["document-edit", "draw-freehand"], QStyle.StandardPixmap.SP_FileDialogDetailedView),
            "Toggle feature reorder/delete mode",
            checkable=True,
        )
        self.feature_edit_btn.toggled.connect(self._toggle_feature_edit)
        header.addWidget(self.feature_edit_btn)

        layout.addLayout(header)

        self.feature_list = ReorderListWidget(group)
        self.feature_list.itemClicked.connect(self._on_feature_item_clicked)
        self.feature_list.order_changed.connect(self._on_feature_order_changed)
        layout.addWidget(self.feature_list)

        return group

    def _build_relation_section(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Editor Pane - Relations", parent)
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.addStretch(1)

        add_btn = self._make_small_button(
            self._icon(["list-add", "plus"], QStyle.StandardPixmap.SP_FileDialogNewFolder),
            "Add relation",
        )
        add_btn.clicked.connect(self._add_relation)
        header.addWidget(add_btn)

        remove_btn = self._make_small_button(
            self._icon(["list-remove", "edit-delete"], QStyle.StandardPixmap.SP_TrashIcon),
            "Remove selected relation",
        )
        remove_btn.clicked.connect(self._remove_selected_relation)
        header.addWidget(remove_btn)

        self.relation_edit_btn = self._make_small_button(
            self._icon(["document-edit", "draw-freehand"], QStyle.StandardPixmap.SP_FileDialogDetailedView),
            "Toggle relation reorder/delete mode",
            checkable=True,
        )
        self.relation_edit_btn.toggled.connect(self._toggle_relation_edit)
        header.addWidget(self.relation_edit_btn)

        layout.addLayout(header)

        self.relation_list = ReorderListWidget(group)
        self.relation_list.itemClicked.connect(self._on_relation_item_clicked)
        self.relation_list.order_changed.connect(self._on_relation_order_changed)
        layout.addWidget(self.relation_list)

        return group

    def _build_detail_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("DetailPanel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        self.detail_title = QLabel("Details", panel)
        self.detail_title.setStyleSheet("font-weight: 700; font-size: 14px;")
        header.addWidget(self.detail_title)
        header.addStretch(1)

        close_btn = self._make_small_button(
            self._icon(["window-close", "dialog-close"], QStyle.StandardPixmap.SP_DockWidgetCloseButton),
            "Close details",
        )
        close_btn.clicked.connect(self._hide_detail)
        header.addWidget(close_btn)

        layout.addLayout(header)

        self.detail_body = QWidget(panel)
        self.detail_body_layout = QVBoxLayout(self.detail_body)
        self.detail_body_layout.setContentsMargins(0, 4, 0, 0)
        self.detail_body_layout.setSpacing(6)
        layout.addWidget(self.detail_body)

        return panel

    def _feature_name_map(self) -> Dict[str, str]:
        return {feature.feature_id: feature.name for feature in self.document.features}

    def _feature_options(self) -> List[DropdownOption]:
        options: List[DropdownOption] = []
        for feature in self.document.features:
            label = f"{feature.name} ({feature.feature_id})"
            search_text = f"{feature.name} {feature.feature_id} {label}".lower()
            options.append(DropdownOption(value=feature.feature_id, label=label, search_text=search_text))
        return options

    def _pretty_kind(self, kind: str) -> str:
        return kind.upper() if kind in {"or", "xor"} else kind.capitalize()

    def _clear_detail_body(self) -> None:
        while self.detail_body_layout.count():
            item = self.detail_body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _hide_detail(self) -> None:
        self.active_detail = None
        self.detail_panel.hide()

    def _show_detail(self) -> None:
        self.detail_panel.show()

    def _refresh_all(self, close_detail: bool) -> None:
        self._refresh_feature_list()
        self._refresh_relation_list()

        if close_detail:
            self._hide_detail()

        self._update_actions()
        self._update_window_title()
        self._render_preview()

    def _update_window_title(self) -> None:
        filename = self.document.json_path.name if self.document.json_path else "untitled"
        dirty = " *" if self.document.dirty else ""
        self.setWindowTitle(f"Feature Diagram Editor - {filename}{dirty}")

    def _update_actions(self) -> None:
        self.action_save.setEnabled(self.document.json_path is not None)
        self.action_reexport.setEnabled(self.document.export_path is not None)
        self.action_undo.setEnabled(self._history_index > 0)
        self.action_redo.setEnabled(self._history_index < len(self._history) - 1)

    def _render_preview(self) -> None:
        try:
            if not self.document.features:
                self.preview.clear_preview()
            else:
                self.preview.render_diagram(self.document.to_core_diagram())
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            self.preview.clear_preview()
            QMessageBox.critical(self, "Preview Error", str(exc))

    def _refresh_feature_list(self) -> None:
        self.feature_list.blockSignals(True)
        self.feature_list.clear()
        self.feature_list.set_reorder_enabled(self.feature_edit_mode)

        for index, feature in enumerate(self.document.features):
            row = ListRowWidget(self.feature_list)
            row.set_content(title=feature.name, subtitle=feature.feature_id)
            row.set_reference_style(feature.is_reference)
            row.set_edit_mode(self.feature_edit_mode)
            if self.feature_edit_mode:
                row.delete_clicked.connect(lambda i=index: self._delete_feature_index(i))
                row.drag_requested.connect(lambda i=index: self.feature_list.begin_drag_for_source(i))
            self.feature_list.add_row(index, row)

        self.feature_list.blockSignals(False)

    def _refresh_relation_list(self) -> None:
        self.relation_list.blockSignals(True)
        self.relation_list.clear()
        self.relation_list.set_reorder_enabled(self.relation_edit_mode)

        name_map = self._feature_name_map()
        for index, relation in enumerate(self.document.relations):
            parent_name = name_map.get(relation.parent, relation.parent)
            child_name = name_map.get(relation.child, relation.child)

            subtitle = f"{parent_name} ({relation.parent}) → {child_name} ({relation.child})"
            meta = f"Group: {relation.group}" if relation.group else ""

            row = ListRowWidget(self.relation_list)
            row.set_content(title=self._pretty_kind(relation.kind), subtitle=subtitle, meta=meta)
            row.set_edit_mode(self.relation_edit_mode)
            if self.relation_edit_mode:
                row.delete_clicked.connect(lambda i=index: self._delete_relation_index(i))
                row.drag_requested.connect(lambda i=index: self.relation_list.begin_drag_for_source(i))
            self.relation_list.add_row(index, row)

        self.relation_list.blockSignals(False)

    def _toggle_feature_edit(self, checked: bool) -> None:
        self.feature_edit_mode = checked
        if checked:
            self._hide_detail()
        self._refresh_feature_list()

    def _toggle_relation_edit(self, checked: bool) -> None:
        self.relation_edit_mode = checked
        if checked:
            self._hide_detail()
        self._refresh_relation_list()

    def _add_feature(self) -> None:
        feature = ask_feature(self, existing_ids=set(self.document.feature_ids), title="Add Feature")
        if feature is None:
            return
        self.document.add_feature(feature)
        self._commit_mutation(close_detail=True)

    def _remove_selected_feature(self) -> None:
        item = self.feature_list.currentItem()
        if item is None:
            return
        index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(index, int):
            return
        self.document.delete_feature_at(index)
        self._commit_mutation(close_detail=True)

    def _delete_feature_index(self, index: int) -> None:
        if not (0 <= index < len(self.document.features)):
            return
        self.document.delete_feature_at(index)
        self._commit_mutation(close_detail=True)

    def _add_relation(self) -> None:
        if not self.document.features:
            QMessageBox.critical(self, "No Features", "Add at least one feature before creating relations.")
            return

        existing_group_ids = {relation.group for relation in self.document.relations if relation.group}
        relations = ask_relation(
            self,
            feature_options=self._feature_options(),
            existing_group_ids=existing_group_ids,
            title="Add Relation",
        )
        if not relations:
            return

        for relation in relations:
            self.document.add_relation(relation)
        self._commit_mutation(close_detail=True)

    def _remove_selected_relation(self) -> None:
        item = self.relation_list.currentItem()
        if item is None:
            return
        index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(index, int):
            return
        self.document.delete_relation_at(index)
        self._commit_mutation(close_detail=True)

    def _delete_relation_index(self, index: int) -> None:
        if not (0 <= index < len(self.document.relations)):
            return
        self.document.delete_relation_at(index)
        self._commit_mutation(close_detail=True)

    def _on_feature_item_clicked(self, item) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(index, int):
            return
        if self.feature_edit_mode:
            return
        self._open_feature_detail(index)

    def _on_relation_item_clicked(self, item) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(index, int):
            return
        if self.relation_edit_mode:
            return
        self._open_relation_detail(index)

    def _on_feature_order_changed(self, order: list) -> None:
        if not self.feature_edit_mode:
            return
        old = list(self.document.features)
        if len(order) != len(old):
            return
        self.document.features = [old[i] for i in order]
        self._commit_mutation(close_detail=True)

    def _on_relation_order_changed(self, order: list) -> None:
        if not self.relation_edit_mode:
            return
        old = list(self.document.relations)
        if len(order) != len(old):
            return
        self.document.relations = [old[i] for i in order]
        self._commit_mutation(close_detail=True)

    def _open_feature_detail(self, index: int) -> None:
        if not (0 <= index < len(self.document.features)):
            return

        feature = self.document.features[index]
        self.active_detail = ("feature", index)

        self._show_detail()
        self._clear_detail_body()
        self.detail_title.setText(f"Feature - {feature.feature_id}")

        panel = QWidget(self.detail_body)
        layout = QVBoxLayout(panel)

        form = QFormLayout()
        name_edit = QLineEdit(feature.name, panel)
        id_edit = QLineEdit(feature.feature_id, panel)
        ref_check = QCheckBox("Reference feature", panel)
        ref_check.setChecked(feature.is_reference)

        form.addRow("Name", name_edit)
        form.addRow("ID", id_edit)
        form.addRow("", ref_check)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        delete_btn = QPushButton("Delete", panel)
        apply_btn = QPushButton("Apply", panel)

        def on_apply() -> None:
            new_id = id_edit.text().strip()
            new_name = name_edit.text().strip() or new_id
            if not new_id:
                QMessageBox.critical(self, "Invalid Feature", "Feature ID cannot be empty.")
                return

            existing = set(self.document.feature_ids)
            existing.discard(feature.feature_id)
            if new_id in existing:
                QMessageBox.critical(self, "Duplicate ID", f"Feature ID '{new_id}' already exists.")
                return

            updated = FeatureItem(feature_id=new_id, name=new_name, is_reference=ref_check.isChecked())
            self.document.update_feature(index, updated)
            self._commit_mutation(close_detail=False)
            new_index = self.document.feature_index_by_id(new_id)
            if new_index is not None:
                self._open_feature_detail(new_index)

        def on_delete() -> None:
            self.document.delete_feature_at(index)
            self._commit_mutation(close_detail=True)

        apply_btn.clicked.connect(on_apply)
        delete_btn.clicked.connect(on_delete)

        buttons.addWidget(apply_btn)
        buttons.addWidget(delete_btn)
        layout.addLayout(buttons)

        self.detail_body_layout.addWidget(panel)

    def _open_relation_detail(self, index: int) -> None:
        if not (0 <= index < len(self.document.relations)):
            return

        relation = self.document.relations[index]
        self.active_detail = ("relation", index)

        self._show_detail()
        self._clear_detail_body()
        self.detail_title.setText(f"Relation - {relation_to_display(relation)}")

        panel = QWidget(self.detail_body)
        layout = QVBoxLayout(panel)

        form = QFormLayout()

        kind_combo = QComboBox(panel)
        kind_combo.addItems(["mandatory", "optional", "or", "xor", "dependency"])
        kind_combo.setCurrentText(relation.kind)

        options = self._feature_options()

        parent_combo = SearchableComboBox(panel)
        parent_combo.set_options(options)
        parent_combo.set_value(relation.parent)

        child_combo = SearchableComboBox(panel)
        child_combo.set_options(options)
        child_combo.set_value(relation.child)

        group_label = QLabel("Group", panel)
        group_edit = QLineEdit(relation.group, panel)

        form.addRow("Type", kind_combo)
        form.addRow("Parent", parent_combo)
        form.addRow("Child", child_combo)
        form.addRow(group_label, group_edit)

        def refresh_group_visibility() -> None:
            visible = kind_combo.currentText().strip().lower() in {"xor", "or"}
            group_label.setVisible(visible)
            group_edit.setVisible(visible)

        kind_combo.currentTextChanged.connect(lambda _text: refresh_group_visibility())
        refresh_group_visibility()

        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)

        delete_btn = QPushButton("Delete", panel)
        apply_btn = QPushButton("Apply", panel)

        def on_apply() -> None:
            kind = kind_combo.currentText().strip().lower()
            parent_id = parent_combo.value()
            child_id = child_combo.value()
            group = group_edit.text().strip()

            if kind not in {"mandatory", "optional", "or", "xor", "dependency"}:
                QMessageBox.critical(self, "Invalid Relation", "Please choose a valid relation type.")
                return
            if parent_id not in self.document.feature_ids:
                QMessageBox.critical(self, "Invalid Relation", "Please choose a valid parent feature.")
                return
            if child_id not in self.document.feature_ids:
                QMessageBox.critical(self, "Invalid Relation", "Please choose a valid child feature.")
                return
            if kind not in {"xor", "or"}:
                group = ""

            updated = RelationItem(kind=kind, parent=parent_id, child=child_id, group=group)
            self.document.update_relation(index, updated)
            self._commit_mutation(close_detail=False)
            if 0 <= index < len(self.document.relations):
                self._open_relation_detail(index)

        def on_delete() -> None:
            self.document.delete_relation_at(index)
            self._commit_mutation(close_detail=True)

        apply_btn.clicked.connect(on_apply)
        delete_btn.clicked.connect(on_delete)

        buttons.addWidget(apply_btn)
        buttons.addWidget(delete_btn)
        layout.addLayout(buttons)

        self.detail_body_layout.addWidget(panel)

    def _confirm_discard_if_dirty(self) -> bool:
        if not self.document.dirty:
            return True
        response = QMessageBox.question(
            self,
            "Unsaved Changes",
            "Discard unsaved changes?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return response == QMessageBox.StandardButton.Yes

    def _load_json(self) -> None:
        if not self._confirm_discard_if_dirty():
            return

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Feature Diagram JSON",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        try:
            self.document.load_json_file(Path(filename))
        except (OSError, ValueError, KeyError, ModelParseError) as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        self._reset_history()
        self._refresh_all(close_detail=True)

    def _save_json(self) -> None:
        if self.document.json_path is None:
            return
        try:
            self.document.save_json_file(self.document.json_path)
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return
        self._mark_clean_history()
        self._refresh_all(close_detail=False)

    def _save_json_as(self) -> None:
        default_name = self.document.json_path.name if self.document.json_path else "diagram.json"
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Feature Diagram JSON",
            default_name,
            "JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        try:
            self.document.save_json_file(Path(filename))
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return

        self._mark_clean_history()
        self._refresh_all(close_detail=False)

    def _export_svg(self) -> None:
        if not self.document.features:
            QMessageBox.critical(self, "No Diagram", "Add features before exporting SVG.")
            return

        default_name = "feature-diagram.svg"
        if self.document.json_path is not None:
            default_name = f"{self.document.json_path.stem}.svg"

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Diagram SVG",
            default_name,
            "SVG Files (*.svg);;All Files (*)",
        )
        if not filename:
            return

        try:
            self.document.export_svg(Path(filename))
        except OSError as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
            return

        self._update_actions()

    def _reexport_svg(self) -> None:
        if self.document.export_path is None:
            return

        try:
            self.document.export_svg(self.document.export_path)
        except OSError as exc:
            QMessageBox.critical(self, "Re-export Error", str(exc))
            return

        self._update_actions()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        if self._initial_split_applied:
            return
        total = self.main_split.width()
        if total > 0:
            editor_width = max(300, total // 3)
            preview_width = max(600, total - editor_width)
            self.main_split.setSizes([editor_width, preview_width])
        self._initial_split_applied = True

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.document.dirty:
            response = QMessageBox.question(
                self,
                "Unsaved Changes",
                "Exit without saving changes?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        super().closeEvent(event)


def run() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setStyle("Fusion")
    app.setStyleSheet(APP_DARK_STYLESHEET)

    window = FeatureDiagramEditorWindow()
    window.show()
    app.exec()
