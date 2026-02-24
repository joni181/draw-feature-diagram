"""Dialogs for adding and editing features/relations (PySide6)."""

from __future__ import annotations

from typing import Optional, Sequence, Set

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from .id_utils import suggest_unique_id
from .state import FeatureItem, RELATION_KINDS, RelationItem
from .widgets import SearchableComboBox


class FeatureDialog(QDialog):
    """Dialog for creating/editing a feature."""

    def __init__(
        self,
        parent: Optional[QWidget],
        existing_ids: Set[str],
        initial: Optional[FeatureItem] = None,
        title: str = "Add Feature",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(420, 190)

        self._existing_ids = set(existing_ids)
        self._initial_id = initial.feature_id if initial else ""
        self._id_touched = initial is not None
        self._result: Optional[FeatureItem] = None

        root = QVBoxLayout(self)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        root.addLayout(form)

        self.name_edit = QLineEdit(initial.name if initial else "", self)
        form.addRow("Name", self.name_edit)

        self.id_edit = QLineEdit(initial.feature_id if initial else "", self)
        form.addRow("ID", self.id_edit)

        self.reference_check = QCheckBox("Reference feature", self)
        self.reference_check.setChecked(initial.is_reference if initial else False)
        form.addRow("", self.reference_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

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
        suggestion = suggest_unique_id(self.name_edit.text(), existing)
        self.id_edit.setText(suggestion)

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


class RelationDialog(QDialog):
    """Dialog for creating/editing a relation."""

    def __init__(
        self,
        parent: Optional[QWidget],
        feature_ids: Sequence[str],
        initial: Optional[RelationItem] = None,
        title: str = "Add Relation",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(460, 250)

        self._feature_ids = list(feature_ids)
        self._result: Optional[RelationItem] = None

        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        root.addLayout(form)

        self.kind_combo = QComboBox(self)
        self.kind_combo.addItems(list(RELATION_KINDS))
        if initial:
            idx = self.kind_combo.findText(initial.kind)
            if idx >= 0:
                self.kind_combo.setCurrentIndex(idx)
        form.addRow("Type", self.kind_combo)

        self.parent_combo = SearchableComboBox(self)
        self.parent_combo.set_options(self._feature_ids)
        self.parent_combo.set_value(initial.parent if initial else "")
        form.addRow("Parent", self.parent_combo)

        self.child_combo = SearchableComboBox(self)
        self.child_combo.set_options(self._feature_ids)
        self.child_combo.set_value(initial.child if initial else "")
        form.addRow("Child", self.child_combo)

        self.group_label = QLabel("Group (xor/or)", self)
        self.group_edit = QLineEdit(initial.group if initial else "", self)

        group_row = QHBoxLayout()
        group_row.setContentsMargins(0, 0, 0, 0)
        group_row.addWidget(self.group_edit)
        group_wrap = QWidget(self)
        group_wrap.setLayout(group_row)

        form.addRow(self.group_label, group_wrap)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.kind_combo.currentTextChanged.connect(self._update_group_visibility)
        self._update_group_visibility(self.kind_combo.currentText())

    def result_relation(self) -> Optional[RelationItem]:
        return self._result

    def _update_group_visibility(self, kind_text: str) -> None:
        show = kind_text.lower() in {"xor", "or"}
        self.group_label.setVisible(show)
        self.group_edit.setVisible(show)

    def _on_accept(self) -> None:
        kind = self.kind_combo.currentText().strip().lower()
        parent_id = self.parent_combo.value()
        child_id = self.child_combo.value()
        group = self.group_edit.text().strip()

        if kind not in RELATION_KINDS:
            QMessageBox.critical(self, "Invalid Relation", "Please select a valid relation type.")
            return
        if parent_id not in self._feature_ids:
            QMessageBox.critical(self, "Invalid Relation", "Please choose a valid parent feature ID.")
            return
        if child_id not in self._feature_ids:
            QMessageBox.critical(self, "Invalid Relation", "Please choose a valid child feature ID.")
            return

        if kind not in {"xor", "or"}:
            group = ""

        self._result = RelationItem(kind=kind, parent=parent_id, child=child_id, group=group)
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
    feature_ids: Sequence[str],
    initial: Optional[RelationItem] = None,
    title: str = "Add Relation",
) -> Optional[RelationItem]:
    dialog = RelationDialog(parent, feature_ids=feature_ids, initial=initial, title=title)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.result_relation()
    return None
