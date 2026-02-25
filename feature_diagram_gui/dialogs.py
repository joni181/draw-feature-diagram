"""Dialogs for adding and editing features/relations (PySide6)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Set

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .id_utils import suggest_unique_id
from .state import FeatureItem, RELATION_KINDS, RelationItem
from .widgets import DropdownOption, SearchableComboBox


class BaseEditorDialog(QDialog):
    """Shared modern dialog shell for editor forms."""

    def __init__(self, parent: Optional[QWidget], title: str, subtitle: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(480, 260)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title_label = QLabel(title, self)
        title_label.setObjectName("DialogTitle")
        root.addWidget(title_label)

        subtitle_label = QLabel(subtitle, self)
        subtitle_label.setObjectName("DialogSubtitle")
        subtitle_label.setWordWrap(True)
        root.addWidget(subtitle_label)

        self.form = QFormLayout()
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        self.form.setHorizontalSpacing(12)
        self.form.setVerticalSpacing(10)
        root.addLayout(self.form)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.accepted.connect(self._on_accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

    def _on_accept(self) -> None:
        raise NotImplementedError


class FeatureDialog(BaseEditorDialog):
    """Dialog for creating/editing a feature."""

    def __init__(
        self,
        parent: Optional[QWidget],
        existing_ids: Set[str],
        initial: Optional[FeatureItem] = None,
        title: str = "Add Feature",
    ) -> None:
        subtitle = "Set a display name and ID. IDs must be unique; names may be duplicates."
        super().__init__(parent=parent, title=title, subtitle=subtitle)

        self._existing_ids = set(existing_ids)
        self._initial_id = initial.feature_id if initial else ""
        self._id_touched = initial is not None
        self._result: Optional[FeatureItem] = None

        self.name_edit = QLineEdit(initial.name if initial else "", self)
        self.id_edit = QLineEdit(initial.feature_id if initial else "", self)
        self.reference_check = QCheckBox("Reference feature (render dashed)", self)
        self.reference_check.setChecked(initial.is_reference if initial else False)

        self.form.addRow("Name", self.name_edit)
        self.form.addRow("ID", self.id_edit)
        self.form.addRow("", self.reference_check)

        self.name_edit.textChanged.connect(self._on_name_changed)
        self.id_edit.textEdited.connect(self._on_id_edited)

        if initial is None:
            self._apply_suggestion()

        self.name_edit.setFocus()

    def result_feature(self) -> Optional[FeatureItem]:
        return self._result

    def _on_name_changed(self) -> None:
        if not self._id_touched:
            self._apply_suggestion()

    def _on_id_edited(self) -> None:
        self._id_touched = True

    def _apply_suggestion(self) -> None:
        existing = set(self._existing_ids)
        if self._initial_id:
            existing.discard(self._initial_id)
        self.id_edit.setText(suggest_unique_id(self.name_edit.text(), existing))

    def _on_accept(self) -> None:
        feature_id = self.id_edit.text().strip()
        name = self.name_edit.text().strip() or feature_id

        if not feature_id:
            QMessageBox.critical(self, "Invalid Feature", "Feature ID cannot be empty.")
            return

        existing = set(self._existing_ids)
        if self._initial_id:
            existing.discard(self._initial_id)
        if feature_id in existing:
            QMessageBox.critical(self, "Duplicate ID", f"Feature ID '{feature_id}' already exists.")
            return

        self._result = FeatureItem(
            feature_id=feature_id,
            name=name,
            is_reference=self.reference_check.isChecked(),
        )
        self.accept()


class RelationDialog(BaseEditorDialog):
    """Dialog for creating/editing a relation."""

    def __init__(
        self,
        parent: Optional[QWidget],
        feature_options: Sequence[DropdownOption],
        existing_group_ids: Set[str],
        initial: Optional[RelationItem] = None,
        title: str = "Add Relation",
    ) -> None:
        subtitle = "Choose relation type and link parent/child features."
        super().__init__(parent=parent, title=title, subtitle=subtitle)
        self.resize(520, 430)

        self._feature_options = list(feature_options)
        self._feature_ids = {option.value for option in self._feature_options}
        self._existing_group_ids = {group_id for group_id in existing_group_ids if group_id}
        self._initial_group = initial.group if initial else ""
        self._group_touched = initial is not None and bool(initial.group)
        self._results: List[RelationItem] = []

        self.kind_combo = QComboBox(self)
        self.kind_combo.addItems(list(RELATION_KINDS))
        if initial:
            idx = self.kind_combo.findText(initial.kind)
            if idx >= 0:
                self.kind_combo.setCurrentIndex(idx)

        self.parent_combo = SearchableComboBox(self)
        self.parent_combo.set_options(self._feature_options)
        self.parent_combo.set_value(initial.parent if initial else "")

        self.child_combo = SearchableComboBox(self)
        self.child_combo.set_options(self._feature_options)
        self.child_combo.set_value(initial.child if initial else "")

        self.group_edit = QLineEdit(initial.group if initial else "", self)
        self.group_edit.textEdited.connect(self._on_group_edited)

        self.child_multi_panel = QWidget(self)
        multi_layout = QVBoxLayout(self.child_multi_panel)
        multi_layout.setContentsMargins(0, 0, 0, 0)
        multi_layout.setSpacing(6)

        self.child_search_edit = QLineEdit(self.child_multi_panel)
        self.child_search_edit.setPlaceholderText("Search children by name or ID...")
        self.child_search_edit.textChanged.connect(self._filter_multi_child_list)
        multi_layout.addWidget(self.child_search_edit)

        self.child_multi_list = QListWidget(self.child_multi_panel)
        self.child_multi_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.child_multi_list.setMinimumHeight(145)
        multi_layout.addWidget(self.child_multi_list)

        for option in self._feature_options:
            item = QListWidgetItem(option.label, self.child_multi_list)
            item.setData(Qt.ItemDataRole.UserRole, option.value)
            item.setData(Qt.ItemDataRole.UserRole + 1, option.search_text)
            if initial and option.value == initial.child:
                item.setSelected(True)

        self.form.addRow("Type", self.kind_combo)
        self.form.addRow("Parent", self.parent_combo)
        self._single_child_label = QLabel("Child", self)
        self.form.addRow(self._single_child_label, self.child_combo)
        self._multi_child_label = QLabel("Children", self)
        self.form.addRow(self._multi_child_label, self.child_multi_panel)
        self.form.addRow("Group (xor/or)", self.group_edit)

        self.kind_combo.currentTextChanged.connect(self._on_kind_changed)
        self.parent_combo.currentIndexChanged.connect(self._on_parent_changed)
        self._update_group_visibility(self.kind_combo.currentText())
        if initial is None:
            self._apply_group_suggestion()

    def result_relations(self) -> List[RelationItem]:
        return list(self._results)

    def _on_kind_changed(self, kind_text: str) -> None:
        self._update_group_visibility(kind_text)
        if kind_text.lower() in {"xor", "or"} and not self._group_touched:
            self._apply_group_suggestion()

    def _on_parent_changed(self) -> None:
        if not self._group_touched and self.kind_combo.currentText().strip().lower() in {"xor", "or"}:
            self._apply_group_suggestion()

    def _on_group_edited(self) -> None:
        self._group_touched = True

    def _update_group_visibility(self, kind_text: str) -> None:
        visible = kind_text.lower() in {"xor", "or"}
        self.group_edit.setVisible(visible)
        label = self.form.labelForField(self.group_edit)
        if label is not None:
            label.setVisible(visible)
        self._single_child_label.setVisible(not visible)
        self.child_combo.setVisible(not visible)
        self._multi_child_label.setVisible(visible)
        self.child_multi_panel.setVisible(visible)

    def _suggest_group_id(self, parent_id: str) -> str:
        base = f"{parent_id}_group" if parent_id else "group"
        existing = set(self._existing_group_ids)
        if self._initial_group:
            existing.discard(self._initial_group)
        if base not in existing:
            return base
        counter = 1
        while True:
            candidate = f"{base}_{counter}"
            if candidate not in existing:
                return candidate
            counter += 1

    def _apply_group_suggestion(self) -> None:
        parent_id = self.parent_combo.value().strip()
        if not parent_id:
            return
        self.group_edit.setText(self._suggest_group_id(parent_id))

    def _filter_multi_child_list(self, query: str) -> None:
        terms = [part for part in query.strip().lower().split() if part]
        for row in range(self.child_multi_list.count()):
            item = self.child_multi_list.item(row)
            search_text = str(item.data(Qt.ItemDataRole.UserRole + 1) or "").lower()
            matches = not terms or all(term in search_text for term in terms)
            item.setHidden(not matches)

    def _selected_child_ids(self) -> List[str]:
        child_ids: List[str] = []
        for item in self.child_multi_list.selectedItems():
            value = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(value, str):
                child_ids.append(value)
        return child_ids

    def _on_accept(self) -> None:
        kind = self.kind_combo.currentText().strip().lower()
        parent_id = self.parent_combo.value()
        group = self.group_edit.text().strip()

        if kind not in RELATION_KINDS:
            QMessageBox.critical(self, "Invalid Relation", "Please select a valid relation type.")
            return
        if parent_id not in self._feature_ids:
            QMessageBox.critical(self, "Invalid Relation", "Please choose a valid parent feature.")
            return

        if kind in {"xor", "or"}:
            child_ids = self._selected_child_ids()
            if not child_ids:
                QMessageBox.critical(
                    self,
                    "Invalid Relation",
                    "Please choose at least one child feature for XOR/OR relations.",
                )
                return
            if not group:
                group = self._suggest_group_id(parent_id)
                self.group_edit.setText(group)
            self._results = [
                RelationItem(kind=kind, parent=parent_id, child=child_id, group=group)
                for child_id in child_ids
            ]
        else:
            child_id = self.child_combo.value()
            if child_id not in self._feature_ids:
                QMessageBox.critical(self, "Invalid Relation", "Please choose a valid child feature.")
                return
            self._results = [RelationItem(kind=kind, parent=parent_id, child=child_id, group="")]
        self.accept()


def ask_feature(
    parent: Optional[QWidget],
    existing_ids: Set[str],
    initial: Optional[FeatureItem] = None,
    title: str = "Add Feature",
) -> Optional[FeatureItem]:
    dialog = FeatureDialog(parent, existing_ids=existing_ids, initial=initial, title=title)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.result_feature()
    return None


def ask_relation(
    parent: Optional[QWidget],
    feature_options: Sequence[DropdownOption],
    existing_group_ids: Set[str],
    initial: Optional[RelationItem] = None,
    title: str = "Add Relation",
) -> Optional[List[RelationItem]]:
    dialog = RelationDialog(
        parent,
        feature_options=feature_options,
        existing_group_ids=existing_group_ids,
        initial=initial,
        title=title,
    )
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.result_relations()
    return None
