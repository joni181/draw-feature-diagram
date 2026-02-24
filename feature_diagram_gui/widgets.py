"""Reusable Qt widgets for the feature diagram GUI."""

from __future__ import annotations

from typing import Iterable, List, Optional

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)


class SearchableComboBox(QComboBox):
    """Combo box with a popup containing an internal search field and list."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._options: List[str] = []
        self._popup: Optional[QFrame] = None
        self._search_edit: Optional[QLineEdit] = None
        self._list_widget: Optional[QListWidget] = None

        self.setEditable(False)
        self.setMaxVisibleItems(14)

    def set_options(self, options: Iterable[str]) -> None:
        self._options = list(options)
        current = self.currentText().strip()

        self.blockSignals(True)
        self.clear()
        self.addItems(self._options)
        self.blockSignals(False)

        if current:
            self.set_value(current)

        if self._popup is not None:
            self._populate_popup_list()

    def set_value(self, value: str) -> None:
        idx = self.findText(value)
        if idx >= 0:
            self.setCurrentIndex(idx)
        elif self.count() > 0:
            self.setCurrentIndex(0)
        else:
            self.setCurrentText(value)

    def value(self) -> str:
        return self.currentText().strip()

    def showPopup(self) -> None:  # noqa: N802
        self._ensure_popup()
        self._populate_popup_list()

        if self._popup is None or self._search_edit is None:
            return

        self._search_edit.clear()

        origin = self.mapToGlobal(QPoint(0, self.height()))
        width = max(self.width(), 240)
        height = 250
        self._popup.setGeometry(origin.x(), origin.y(), width, height)
        self._popup.show()
        self._popup.raise_()
        self._search_edit.setFocus()

    def hidePopup(self) -> None:  # noqa: N802
        if self._popup is not None:
            self._popup.hide()

    def _ensure_popup(self) -> None:
        if self._popup is not None:
            return

        popup = QFrame(self, Qt.WindowType.Popup)
        popup.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(popup)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        search_edit = QLineEdit(popup)
        search_edit.setPlaceholderText("Search feature ID...")
        layout.addWidget(search_edit)

        list_widget = QListWidget(popup)
        list_widget.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(list_widget)

        search_edit.textChanged.connect(self._filter_popup_list)
        list_widget.itemClicked.connect(self._on_popup_item_clicked)
        list_widget.itemActivated.connect(self._on_popup_item_clicked)

        self._popup = popup
        self._search_edit = search_edit
        self._list_widget = list_widget

    def _populate_popup_list(self) -> None:
        if self._list_widget is None:
            return

        selected_text = self.currentText().strip()

        self._list_widget.clear()
        for option in self._options:
            self._list_widget.addItem(QListWidgetItem(option))

        if selected_text:
            matches = self._list_widget.findItems(selected_text, Qt.MatchFlag.MatchExactly)
            if matches:
                self._list_widget.setCurrentItem(matches[0])

    def _filter_popup_list(self, query: str) -> None:
        if self._list_widget is None:
            return

        query_norm = query.strip().lower()

        self._list_widget.clear()
        for option in self._options:
            if not query_norm or query_norm in option.lower():
                self._list_widget.addItem(QListWidgetItem(option))

    def _on_popup_item_clicked(self, item: QListWidgetItem) -> None:
        value = item.text()
        self.setCurrentText(value)
        self.hidePopup()


class ReorderTreeWidget(QTreeWidget):
    """Tree widget emitting the row order after internal drag/drop operations."""

    order_changed = Signal(list)

    def dropEvent(self, event) -> None:  # noqa: N802
        super().dropEvent(event)
        order: List[int] = []
        for row in range(self.topLevelItemCount()):
            item = self.topLevelItem(row)
            index = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(index, int):
                order.append(index)
        self.order_changed.emit(order)


def build_tree_item(row_index: int, columns: List[str], draggable: bool) -> QTreeWidgetItem:
    """Build a tree row item with stable source index metadata."""
    item = QTreeWidgetItem(columns)
    item.setData(0, Qt.ItemDataRole.UserRole, row_index)

    flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
    if draggable:
        flags |= Qt.ItemFlag.ItemIsDragEnabled | Qt.ItemFlag.ItemIsDropEnabled
    item.setFlags(flags)
    return item
