"""PySide6 desktop editor with live preview for feature diagrams."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
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
    QTreeWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QCheckBox,
)

from feature_diagram_core.models import ModelParseError

from .dialogs import ask_feature, ask_relation
from .preview import DiagramGraphicsView
from .state import DiagramDocument, FeatureItem, RelationItem, relation_to_display
from .widgets import ReorderTreeWidget, SearchableComboBox, build_tree_item


class FeatureDiagramEditorWindow(QMainWindow):
    """Main desktop editor window."""

    def __init__(self) -> None:
        super().__init__()

        self.document = DiagramDocument()
        self.feature_edit_mode = False
        self.relation_edit_mode = False
        self.active_detail: Optional[Tuple[str, int]] = None

        self.setWindowTitle("Feature Diagram Editor")
        self.resize(1500, 920)
        self.setMinimumSize(1140, 740)

        self._build_ui()
        self._build_toolbar()

        self.statusBar().showMessage(
            "Preview controls: left click zoom in, right click zoom out, middle drag pan, ctrl/cmd+wheel zoom"
        )

        self._refresh_all(close_detail=True)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)

        split = QSplitter(Qt.Orientation.Horizontal, root)
        root_layout.addWidget(split)

        editor_shell = QWidget(split)
        editor_layout = QVBoxLayout(editor_shell)
        editor_layout.setContentsMargins(0, 0, 0, 0)

        self.editor_split = QSplitter(Qt.Orientation.Vertical, editor_shell)
        editor_layout.addWidget(self.editor_split)

        lists_widget = QWidget(self.editor_split)
        lists_layout = QVBoxLayout(lists_widget)
        lists_layout.setContentsMargins(0, 0, 0, 0)

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
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 4)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("File", self)
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        style = self.style()

        self.action_load = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton), "Load", self)
        self.action_load.triggered.connect(self._load_json)
        toolbar.addAction(self.action_load)

        self.action_save = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton), "Save", self)
        self.action_save.triggered.connect(self._save_json)
        toolbar.addAction(self.action_save)

        self.action_save_as = QAction(style.standardIcon(QStyle.StandardPixmap.SP_DriveFDIcon), "Save As", self)
        self.action_save_as.triggered.connect(self._save_json_as)
        toolbar.addAction(self.action_save_as)

        toolbar.addSeparator()

        self.action_export = QAction(style.standardIcon(QStyle.StandardPixmap.SP_ArrowDown), "Export", self)
        self.action_export.triggered.connect(self._export_svg)
        toolbar.addAction(self.action_export)

        self.action_reexport = QAction(style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload), "Re-export", self)
        self.action_reexport.triggered.connect(self._reexport_svg)
        toolbar.addAction(self.action_reexport)

    def _build_feature_section(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Editor Pane - Features", parent)
        layout = QVBoxLayout(group)

        header = QHBoxLayout()
        header.addStretch(1)

        add_btn = QToolButton(group)
        add_btn.setText("+")
        add_btn.setToolTip("Add feature")
        add_btn.clicked.connect(self._add_feature)
        header.addWidget(add_btn)

        remove_btn = QToolButton(group)
        remove_btn.setText("−")
        remove_btn.setToolTip("Remove selected feature")
        remove_btn.clicked.connect(self._remove_selected_feature)
        header.addWidget(remove_btn)

        self.feature_edit_btn = QToolButton(group)
        self.feature_edit_btn.setText("✎")
        self.feature_edit_btn.setToolTip("Toggle feature edit mode")
        self.feature_edit_btn.setCheckable(True)
        self.feature_edit_btn.toggled.connect(self._toggle_feature_edit)
        header.addWidget(self.feature_edit_btn)

        layout.addLayout(header)

        tree = ReorderTreeWidget(group)
        tree.setColumnCount(5)
        tree.setHeaderLabels(["", "ID", "Name", "Type", ""])
        tree.setRootIsDecorated(False)
        tree.setUniformRowHeights(True)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        header_view = tree.header()
        header_view.setStretchLastSection(False)
        header_view.resizeSection(0, 34)
        header_view.resizeSection(1, 160)
        header_view.resizeSection(2, 250)
        header_view.resizeSection(3, 60)
        header_view.resizeSection(4, 42)

        tree.itemClicked.connect(self._on_feature_item_clicked)
        tree.order_changed.connect(self._on_feature_order_changed)

        self.feature_tree = tree
        layout.addWidget(tree)
        return group

    def _build_relation_section(self, parent: QWidget) -> QGroupBox:
        group = QGroupBox("Editor Pane - Relations", parent)
        layout = QVBoxLayout(group)

        header = QHBoxLayout()
        header.addStretch(1)

        add_btn = QToolButton(group)
        add_btn.setText("+")
        add_btn.setToolTip("Add relation")
        add_btn.clicked.connect(self._add_relation)
        header.addWidget(add_btn)

        remove_btn = QToolButton(group)
        remove_btn.setText("−")
        remove_btn.setToolTip("Remove selected relation")
        remove_btn.clicked.connect(self._remove_selected_relation)
        header.addWidget(remove_btn)

        self.relation_edit_btn = QToolButton(group)
        self.relation_edit_btn.setText("✎")
        self.relation_edit_btn.setToolTip("Toggle relation edit mode")
        self.relation_edit_btn.setCheckable(True)
        self.relation_edit_btn.toggled.connect(self._toggle_relation_edit)
        header.addWidget(self.relation_edit_btn)

        layout.addLayout(header)

        tree = ReorderTreeWidget(group)
        tree.setColumnCount(6)
        tree.setHeaderLabels(["", "Type", "Parent", "Child", "Group", ""])
        tree.setRootIsDecorated(False)
        tree.setUniformRowHeights(True)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

        header_view = tree.header()
        header_view.setStretchLastSection(False)
        header_view.resizeSection(0, 34)
        header_view.resizeSection(1, 95)
        header_view.resizeSection(2, 140)
        header_view.resizeSection(3, 140)
        header_view.resizeSection(4, 100)
        header_view.resizeSection(5, 42)

        tree.itemClicked.connect(self._on_relation_item_clicked)
        tree.order_changed.connect(self._on_relation_order_changed)

        self.relation_tree = tree
        layout.addWidget(tree)
        return group

    def _build_detail_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)

        header = QHBoxLayout()
        self.detail_title = QLabel("Details", panel)
        self.detail_title.setStyleSheet("font-weight: 600;")
        header.addWidget(self.detail_title)
        header.addStretch(1)

        close_btn = QToolButton(panel)
        close_btn.setText("X")
        close_btn.setToolTip("Close details")
        close_btn.clicked.connect(self._hide_detail)
        header.addWidget(close_btn)

        layout.addLayout(header)

        self.detail_body = QWidget(panel)
        self.detail_body_layout = QVBoxLayout(self.detail_body)
        self.detail_body_layout.setContentsMargins(0, 4, 0, 0)
        layout.addWidget(self.detail_body)

        return panel

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
        self._refresh_feature_tree()
        self._refresh_relation_tree()

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

    def _render_preview(self) -> None:
        try:
            if not self.document.features:
                self.preview.clear_preview()
            else:
                self.preview.render_diagram(self.document.to_core_diagram())
        except Exception as exc:  # pragma: no cover - defensive UI fallback
            self.preview.clear_preview()
            QMessageBox.critical(self, "Preview Error", str(exc))

    def _configure_drag_mode(self, tree: QTreeWidget, enabled: bool) -> None:
        if enabled:
            tree.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            tree.setDragEnabled(True)
            tree.setAcceptDrops(True)
            tree.setDropIndicatorShown(True)
            tree.setDefaultDropAction(Qt.DropAction.MoveAction)
        else:
            tree.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
            tree.setDragEnabled(False)
            tree.setAcceptDrops(False)
            tree.setDropIndicatorShown(False)

    def _refresh_feature_tree(self) -> None:
        self.feature_tree.blockSignals(True)
        self.feature_tree.clear()
        self._configure_drag_mode(self.feature_tree, self.feature_edit_mode)

        for index, feature in enumerate(self.document.features):
            item = build_tree_item(
                index,
                [
                    "≡" if self.feature_edit_mode else "",
                    feature.feature_id,
                    feature.name,
                    "ref" if feature.is_reference else "std",
                    "🗑" if self.feature_edit_mode else "",
                ],
                draggable=self.feature_edit_mode,
            )
            self.feature_tree.addTopLevelItem(item)

        self.feature_tree.blockSignals(False)

    def _refresh_relation_tree(self) -> None:
        self.relation_tree.blockSignals(True)
        self.relation_tree.clear()
        self._configure_drag_mode(self.relation_tree, self.relation_edit_mode)

        for index, relation in enumerate(self.document.relations):
            item = build_tree_item(
                index,
                [
                    "≡" if self.relation_edit_mode else "",
                    relation.kind,
                    relation.parent,
                    relation.child,
                    relation.group if relation.group else "-",
                    "🗑" if self.relation_edit_mode else "",
                ],
                draggable=self.relation_edit_mode,
            )
            self.relation_tree.addTopLevelItem(item)

        self.relation_tree.blockSignals(False)

    def _toggle_feature_edit(self, checked: bool) -> None:
        self.feature_edit_mode = checked
        self.feature_edit_btn.setText("✓" if checked else "✎")
        if checked:
            self._hide_detail()
        self._refresh_feature_tree()

    def _toggle_relation_edit(self, checked: bool) -> None:
        self.relation_edit_mode = checked
        self.relation_edit_btn.setText("✓" if checked else "✎")
        if checked:
            self._hide_detail()
        self._refresh_relation_tree()

    def _add_feature(self) -> None:
        feature = ask_feature(self, existing_ids=set(self.document.feature_ids), title="Add Feature")
        if feature is None:
            return
        self.document.add_feature(feature)
        self._refresh_all(close_detail=True)

    def _remove_selected_feature(self) -> None:
        item = self.feature_tree.currentItem()
        if item is None:
            return
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(index, int):
            return
        self.document.delete_feature_at(index)
        self._refresh_all(close_detail=True)

    def _add_relation(self) -> None:
        if not self.document.features:
            QMessageBox.critical(self, "No Features", "Add at least one feature before creating relations.")
            return
        relation = ask_relation(self, feature_ids=self.document.feature_ids, title="Add Relation")
        if relation is None:
            return
        self.document.add_relation(relation)
        self._refresh_all(close_detail=True)

    def _remove_selected_relation(self) -> None:
        item = self.relation_tree.currentItem()
        if item is None:
            return
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(index, int):
            return
        self.document.delete_relation_at(index)
        self._refresh_all(close_detail=True)

    def _on_feature_item_clicked(self, item, column: int) -> None:
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(index, int):
            return

        if self.feature_edit_mode:
            if column == 4:
                self.document.delete_feature_at(index)
                self._refresh_all(close_detail=True)
            return

        self._open_feature_detail(index)

    def _on_relation_item_clicked(self, item, column: int) -> None:
        index = item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(index, int):
            return

        if self.relation_edit_mode:
            if column == 5:
                self.document.delete_relation_at(index)
                self._refresh_all(close_detail=True)
            return

        self._open_relation_detail(index)

    def _on_feature_order_changed(self, order: list) -> None:
        if not self.feature_edit_mode:
            return
        old = list(self.document.features)
        if len(order) != len(old):
            return
        self.document.features = [old[i] for i in order]
        self.document.set_dirty(True)
        self._refresh_all(close_detail=True)

    def _on_relation_order_changed(self, order: list) -> None:
        if not self.relation_edit_mode:
            return
        old = list(self.document.relations)
        if len(order) != len(old):
            return
        self.document.relations = [old[i] for i in order]
        self.document.set_dirty(True)
        self._refresh_all(close_detail=True)

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
            self._refresh_all(close_detail=False)
            new_index = self.document.feature_index_by_id(new_id)
            if new_index is not None:
                self._open_feature_detail(new_index)

        def on_delete() -> None:
            self.document.delete_feature_at(index)
            self._refresh_all(close_detail=True)

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

        parent_combo = SearchableComboBox(panel)
        parent_combo.set_options(self.document.feature_ids)
        parent_combo.set_value(relation.parent)

        child_combo = SearchableComboBox(panel)
        child_combo.set_options(self.document.feature_ids)
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
                QMessageBox.critical(self, "Invalid Relation", "Please choose a valid parent feature ID.")
                return
            if child_id not in self.document.feature_ids:
                QMessageBox.critical(self, "Invalid Relation", "Please choose a valid child feature ID.")
                return
            if kind not in {"xor", "or"}:
                group = ""

            updated = RelationItem(kind=kind, parent=parent_id, child=child_id, group=group)
            self.document.update_relation(index, updated)
            self._refresh_all(close_detail=False)
            if 0 <= index < len(self.document.relations):
                self._open_relation_detail(index)

        def on_delete() -> None:
            self.document.delete_relation_at(index)
            self._refresh_all(close_detail=True)

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

        self._refresh_all(close_detail=True)

    def _save_json(self) -> None:
        if self.document.json_path is None:
            return
        try:
            self.document.save_json_file(self.document.json_path)
        except OSError as exc:
            QMessageBox.critical(self, "Save Error", str(exc))
            return
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

    window = FeatureDiagramEditorWindow()
    window.show()
    app.exec()
