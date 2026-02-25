"""Reusable Qt widgets for the feature diagram GUI."""

from __future__ import annotations

import html
from dataclasses import dataclass
from typing import Iterable, List, Optional

from PySide6.QtCore import QPoint, QEvent, QTimer, Qt, Signal
from PySide6.QtGui import QDrag, QMouseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class DropdownOption:
    """Value/label/search triple for searchable dropdown entries."""

    value: str
    label: str
    search_text: str


class SearchableComboBox(QComboBox):
    """Combo box with in-popup search and highlighted matching results."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self._options: List[DropdownOption] = []
        self._popup: Optional[QFrame] = None
        self._search_line: Optional[QLineEdit] = None
        self._list_widget: Optional[QListWidget] = None

        self.setEditable(False)
        self.setMaxVisibleItems(14)

    def set_options(self, options: Iterable[DropdownOption]) -> None:
        current_value = self.value()
        self._options = list(options)

        self.blockSignals(True)
        self.clear()
        for option in self._options:
            self.addItem(option.label, option.value)
        self.blockSignals(False)

        if current_value:
            self.set_value(current_value)
        elif self.count() > 0:
            self.setCurrentIndex(0)

        if self._popup is not None:
            self._populate_popup_list()

    def set_value(self, value: str) -> None:
        for idx, option in enumerate(self._options):
            if option.value == value:
                self.setCurrentIndex(idx)
                return
        if self.count() > 0:
            self.setCurrentIndex(0)

    def value(self) -> str:
        idx = self.currentIndex()
        if 0 <= idx < len(self._options):
            return self._options[idx].value
        return ""

    def showPopup(self) -> None:  # noqa: N802
        self._ensure_popup()
        self._populate_popup_list()

        if self._popup is None or self._search_line is None:
            return

        self._search_line.clear()

        origin = self.mapToGlobal(QPoint(0, self.height() + 2))
        width = max(self.width(), 280)
        self._popup.setGeometry(origin.x(), origin.y(), width, 280)
        self._popup.show()
        self._popup.raise_()
        self._search_line.setFocus()

    def hidePopup(self) -> None:  # noqa: N802
        if self._popup is not None:
            self._popup.hide()

    def eventFilter(self, watched, event) -> bool:  # noqa: A003
        if watched == self._search_line and event.type() == QEvent.Type.KeyPress and self._list_widget is not None:
            key = event.key()
            if key in {Qt.Key.Key_Down, Qt.Key.Key_Up}:
                row = self._list_widget.currentRow()
                if row < 0:
                    row = 0
                row += 1 if key == Qt.Key.Key_Down else -1
                row = max(0, min(self._list_widget.count() - 1, row))
                self._list_widget.setCurrentRow(row)
                return True
            if key in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                self._select_current_popup_item()
                return True
        return super().eventFilter(watched, event)

    def _ensure_popup(self) -> None:
        if self._popup is not None:
            return

        popup = QFrame(self, Qt.WindowType.Popup)
        popup.setObjectName("SearchPopup")

        layout = QVBoxLayout(popup)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        hint = QLabel("Search", popup)
        hint.setObjectName("PopupHint")
        layout.addWidget(hint)

        search_line = QLineEdit(popup)
        search_line.setPlaceholderText("Type to filter by name or ID...")
        layout.addWidget(search_line)

        list_widget = QListWidget(popup)
        list_widget.setObjectName("SearchResults")
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        list_widget.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        layout.addWidget(list_widget)

        search_line.textChanged.connect(self._filter_popup_list)
        search_line.returnPressed.connect(self._select_current_popup_item)
        list_widget.itemClicked.connect(self._on_popup_item_clicked)
        list_widget.itemActivated.connect(self._on_popup_item_clicked)

        search_line.installEventFilter(self)

        self._popup = popup
        self._search_line = search_line
        self._list_widget = list_widget

    def _populate_popup_list(self) -> None:
        query = self._search_line.text() if self._search_line is not None else ""
        self._filter_popup_list(query)

    def _filter_popup_list(self, query: str) -> None:
        if self._list_widget is None:
            return

        self._list_widget.clear()
        terms = [part for part in query.strip().lower().split() if part]

        current_value = self.value()
        first_row = -1

        for option in self._options:
            if terms and not all(term in option.search_text for term in terms):
                continue

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, option.value)

            row_widget = QWidget(self._list_widget)
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(10, 7, 10, 7)
            row_layout.setSpacing(0)

            label = QLabel(self._highlight_matches(option.label, terms), row_widget)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(False)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            row_layout.addWidget(label)

            item.setSizeHint(row_widget.sizeHint())
            self._list_widget.addItem(item)
            self._list_widget.setItemWidget(item, row_widget)

            if first_row < 0:
                first_row = self._list_widget.count() - 1
            if option.value == current_value:
                self._list_widget.setCurrentRow(self._list_widget.count() - 1)

        if self._list_widget.currentRow() < 0 and first_row >= 0:
            self._list_widget.setCurrentRow(first_row)

    def _highlight_matches(self, label: str, terms: List[str]) -> str:
        if not terms:
            return html.escape(label)

        normalized_terms = sorted(set(terms), key=len, reverse=True)
        lower_label = label.lower()
        spans: List[tuple[int, int]] = []

        for term in normalized_terms:
            start = 0
            while True:
                index = lower_label.find(term, start)
                if index < 0:
                    break
                spans.append((index, index + len(term)))
                start = index + len(term)

        if not spans:
            return html.escape(label)

        spans.sort()
        merged: List[tuple[int, int]] = []
        for start, end in spans:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))

        result_parts: List[str] = []
        cursor = 0
        for start, end in merged:
            if cursor < start:
                result_parts.append(html.escape(label[cursor:start]))
            result_parts.append(f"<b>{html.escape(label[start:end])}</b>")
            cursor = end
        if cursor < len(label):
            result_parts.append(html.escape(label[cursor:]))

        return "".join(result_parts)

    def _select_current_popup_item(self) -> None:
        if self._list_widget is None:
            return
        item = self._list_widget.currentItem()
        if item is None:
            return
        self._on_popup_item_clicked(item)

    def _on_popup_item_clicked(self, item: QListWidgetItem) -> None:
        value = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(value, str):
            self.set_value(value)
        self.hidePopup()


class DragHandleLabel(QLabel):
    """Drag handle that emits a signal after a natural drag threshold."""

    drag_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("≡", parent)
        self._press_pos: Optional[QPoint] = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setObjectName("RowHandle")
        self.setFixedWidth(18)

    def _safe_set_cursor(self, cursor: Qt.CursorShape) -> None:
        try:
            self.setCursor(cursor)
        except RuntimeError:
            # Qt may still deliver events to a Python wrapper after C++ deletion.
            pass

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self._safe_set_cursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        try:
            super().mousePressEvent(event)
        except RuntimeError:
            pass

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._press_pos is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position().toPoint() - self._press_pos
            if delta.manhattanLength() >= QApplication.startDragDistance():
                self._press_pos = None
                self.drag_requested.emit()
                event.accept()
                return
        try:
            super().mouseMoveEvent(event)
        except RuntimeError:
            pass

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._press_pos = None
        self._safe_set_cursor(Qt.CursorShape.OpenHandCursor)
        try:
            super().mouseReleaseEvent(event)
        except RuntimeError:
            pass


class ListRowWidget(QFrame):
    """List-row widget with optional drag handle and delete button."""

    delete_clicked = Signal()
    drag_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("ListRow")
        self.setProperty("reference", False)

        root = QHBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(8)

        self.handle = DragHandleLabel(self)
        self.handle.setVisible(False)
        self.handle.drag_requested.connect(self.drag_requested.emit)
        root.addWidget(self.handle)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        self.title_label = QLabel("", self)
        self.title_label.setObjectName("RowTitle")
        self.title_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        text_layout.addWidget(self.title_label)

        self.subtitle_label = QLabel("", self)
        self.subtitle_label.setObjectName("RowSubtitle")
        self.subtitle_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        text_layout.addWidget(self.subtitle_label)

        self.meta_label = QLabel("", self)
        self.meta_label.setObjectName("RowMeta")
        self.meta_label.setVisible(False)
        self.meta_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        text_layout.addWidget(self.meta_label)

        root.addLayout(text_layout, 1)

        self.delete_button = QToolButton(self)
        self.delete_button.setObjectName("RowDelete")
        self.delete_button.setText("🗑")
        self.delete_button.setAutoRaise(True)
        self.delete_button.setVisible(False)
        self.delete_button.clicked.connect(self.delete_clicked.emit)
        root.addWidget(self.delete_button)

    def set_content(self, title: str, subtitle: str = "", meta: str = "") -> None:
        self.title_label.setText(title)
        self.subtitle_label.setVisible(bool(subtitle))
        self.subtitle_label.setText(subtitle)
        self.meta_label.setVisible(bool(meta))
        self.meta_label.setText(meta)

    def set_reference_style(self, enabled: bool) -> None:
        self.setProperty("reference", enabled)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def set_edit_mode(self, enabled: bool) -> None:
        self.handle.setVisible(enabled)
        self.delete_button.setVisible(enabled)


class ReorderListWidget(QListWidget):
    """List widget with handle-only internal reorder and clear drop marker."""

    order_changed = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._reorder_enabled = False
        self._manual_hotspot: Optional[QPoint] = None

        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSpacing(0)
        self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.setDropIndicatorShown(False)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)

        self._drop_line = QFrame(self.viewport())
        self._drop_line.setObjectName("ListDropLine")
        self._drop_line.setFrameShape(QFrame.Shape.HLine)
        self._drop_line.setLineWidth(2)
        self._drop_line.setStyleSheet("background: #7aa7e2; border: none;")
        self._drop_line.hide()

    def set_reorder_enabled(self, enabled: bool) -> None:
        self._reorder_enabled = enabled
        if enabled:
            self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
            self.setDragEnabled(False)
            self.setAcceptDrops(True)
            self.setDropIndicatorShown(False)
            self.setDefaultDropAction(Qt.DropAction.MoveAction)
        else:
            self.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
            self.setDragEnabled(False)
            self.setAcceptDrops(False)
            self.setDropIndicatorShown(False)
            self._drop_line.hide()

    def add_row(self, source_index: int, row_widget: ListRowWidget) -> None:
        item = QListWidgetItem(self)
        item.setData(Qt.ItemDataRole.UserRole, source_index)
        item.setSizeHint(row_widget.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, row_widget)

    def begin_drag_for_source(self, source_index: int) -> None:
        if not self._reorder_enabled:
            return

        row = self._row_for_source(source_index)
        if row < 0:
            return

        self.setCurrentRow(row)
        item = self.item(row)
        widget = self.itemWidget(item)
        hotspot_y = 0
        if widget is not None:
            hotspot_y = max(0, widget.height() // 2)
        self._manual_hotspot = QPoint(14, hotspot_y)
        self.startDrag(Qt.DropAction.MoveAction)

    def startDrag(self, supportedActions) -> None:  # noqa: N802
        if not self._reorder_enabled:
            return

        item = self.currentItem()
        if item is None:
            return

        drag = QDrag(self)
        mime_data = self.model().mimeData(self.selectedIndexes())
        if mime_data is None:
            return

        drag.setMimeData(mime_data)

        row_widget = self.itemWidget(item)
        if row_widget is not None:
            pixmap = row_widget.grab()
            drag.setPixmap(pixmap)

            hotspot = self._manual_hotspot or QPoint(12, pixmap.height() // 2)
            hotspot.setX(max(0, min(pixmap.width() - 1, hotspot.x())))
            hotspot.setY(max(0, min(pixmap.height() - 1, hotspot.y())))
            drag.setHotSpot(hotspot)

        drag.exec(Qt.DropAction.MoveAction)
        self._manual_hotspot = None
        self._drop_line.hide()

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if self._reorder_enabled:
            event.acceptProposedAction()
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._reorder_enabled:
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            target_row = self._insertion_row_from_pos(pos)
            self._show_drop_line(target_row)
            event.acceptProposedAction()
        super().dragMoveEvent(event)

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._drop_line.hide()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        super().dropEvent(event)
        self._drop_line.hide()

        order: List[int] = []
        for row in range(self.count()):
            item = self.item(row)
            source_index = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(source_index, int):
                order.append(source_index)
        # Emit after drag cleanup; immediate refresh can delete active drag widgets.
        QTimer.singleShot(0, lambda new_order=order: self.order_changed.emit(new_order))

    def resizeEvent(self, event) -> None:  # noqa: N802
        if self._drop_line.isVisible():
            self._show_drop_line(self._insertion_row_from_pos(QPoint(0, self._drop_line.y())))
        super().resizeEvent(event)

    def _row_for_source(self, source_index: int) -> int:
        for row in range(self.count()):
            item = self.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == source_index:
                return row
        return -1

    def _insertion_row_from_pos(self, pos: QPoint) -> int:
        for row in range(self.count()):
            rect = self.visualItemRect(self.item(row))
            if pos.y() < rect.center().y():
                return row
        return self.count()

    def _show_drop_line(self, insertion_row: int) -> None:
        if self.count() == 0:
            y = 2
        elif insertion_row <= 0:
            y = self.visualItemRect(self.item(0)).top()
        elif insertion_row >= self.count():
            y = self.visualItemRect(self.item(self.count() - 1)).bottom() + 1
        else:
            y = self.visualItemRect(self.item(insertion_row)).top()

        self._drop_line.setGeometry(8, y - 1, max(24, self.viewport().width() - 16), 2)
        self._drop_line.show()
        self._drop_line.raise_()
