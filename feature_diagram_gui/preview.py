"""PySide6 preview widget for feature diagrams with zoom and pan."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QNativeGestureEvent,
    QPainter,
    QPen,
    QPolygonF,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QGraphicsScene, QGraphicsSimpleTextItem, QGraphicsView, QWidget

from feature_diagram_core.constants import (
    BOX_H,
    DEP_DASH,
    DEP_LANE_SPACING,
    DEP_MARGIN,
    HUB_SPAN_THRESHOLD,
    MARKER_OFFSET,
    MARKER_RADIUS,
    PADDING,
    TRIANGLE_GAP,
)
from feature_diagram_core.geometry import anchor_with_stub, segment_intersects_rect
from feature_diagram_core.layout import FeatureDiagramLayoutEngine
from feature_diagram_core.models import FeatureDiagram


class DiagramGraphicsView(QGraphicsView):
    """Interactive preview view for feature diagrams."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        self.setBackgroundBrush(QColor("#f6f8fb"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        self._zoom_level = 1.0
        self._panning = False
        self._pan_last_pos = None

        self._draw_placeholder("No diagram loaded.")

    def clear_preview(self) -> None:
        self.resetTransform()
        self._zoom_level = 1.0
        self._draw_placeholder("No diagram loaded.")

    def _draw_placeholder(self, text: str) -> None:
        self._scene.clear()
        item = QGraphicsSimpleTextItem(text)
        item.setBrush(QBrush(QColor("#59697a")))
        font = QFont("Helvetica", 12)
        item.setFont(font)
        self._scene.addItem(item)

        bounds = item.boundingRect()
        item.setPos(-bounds.width() / 2, -bounds.height() / 2)
        self._scene.setSceneRect(QRectF(-220, -120, 440, 240))
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def render_diagram(self, diagram: FeatureDiagram) -> None:
        if not diagram.features:
            self.clear_preview()
            return

        self._scene.clear()

        layout = FeatureDiagramLayoutEngine().compute(diagram)
        features = diagram.features
        relations = diagram.relations
        positions = layout.positions
        width_map = layout.width_map
        group_map = layout.group_map

        min_x = min(cx - width_map[fid] / 2 for fid, (cx, _) in positions.items())
        max_x = max(cx + width_map[fid] / 2 for fid, (cx, _) in positions.items())
        min_y = min(cy for _, cy in positions.values()) - BOX_H / 2
        max_y = max(cy for _, cy in positions.values()) + BOX_H / 2

        # Boxes and labels
        for feature_id, feature in features.items():
            cx, cy = positions[feature_id]
            width = width_map[feature_id]
            x0 = cx - width / 2
            y0 = cy - BOX_H / 2

            pen = QPen(QColor("black"), 1.5)
            if feature.is_reference:
                pen.setStyle(Qt.PenStyle.CustomDashLine)
                pen.setDashPattern([6.0, 4.0])

            self._scene.addRect(x0, y0, width, BOX_H, pen, QBrush(QColor("white")))
            self._draw_centered_text(cx, cy, feature.name)

        # Mandatory / optional markers (with optional hub line for wide spans)
        mo_relations = [relation for relation in relations if relation.kind in {"mandatory", "optional"}]
        mo_children: Dict[str, List[str]] = {}
        for relation in mo_relations:
            mo_children.setdefault(relation.parent, []).append(relation.child)

        hub_y_map: Dict[str, float] = {}
        for parent, children in mo_children.items():
            if len(children) < 2:
                continue
            xs = [positions[child][0] for child in children]
            span = max(xs) - min(xs)
            if span >= HUB_SPAN_THRESHOLD:
                py = positions[parent][1]
                parent_bottom = py + BOX_H / 2
                child_tops = [positions[child][1] - BOX_H / 2 for child in children]
                avg_child_top = sum(child_tops) / len(child_tops)
                hub_y_map[parent] = parent_bottom + (avg_child_top - parent_bottom) * 0.5

        for parent, children in mo_children.items():
            use_hub = parent in hub_y_map
            hub_y = hub_y_map.get(parent, 0.0)
            if use_hub:
                xs = [positions[child][0] for child in children]
                line_start = (min(xs), hub_y)
                line_end = (max(xs), hub_y)
                self._draw_line(line_start[0], line_start[1], line_end[0], line_end[1], width=1.4)
                px, py = positions[parent]
                parent_bottom = (px, py + BOX_H / 2)
                self._draw_line(parent_bottom[0], parent_bottom[1], px, hub_y, width=1.4)
                min_x = min(min_x, line_start[0], line_end[0], parent_bottom[0])
                max_x = max(max_x, line_start[0], line_end[0], parent_bottom[0])
                min_y = min(min_y, hub_y, parent_bottom[1])
                max_y = max(max_y, hub_y, parent_bottom[1])

            for relation in mo_relations:
                if relation.parent != parent:
                    continue
                cx, cy = positions[relation.child]
                child_top = cy - BOX_H / 2
                marker_cy = child_top - MARKER_OFFSET
                marker_cx = cx

                if use_hub:
                    self._draw_line(marker_cx, hub_y, marker_cx, marker_cy, width=1.4)
                else:
                    px, py = positions[parent]
                    start = (px, py + BOX_H / 2)
                    self._draw_line(start[0], start[1], marker_cx, marker_cy, width=1.4)

                self._draw_line(marker_cx, marker_cy, cx, child_top, width=1.4)
                fill_color = QColor("black") if relation.kind == "mandatory" else QColor("white")
                circle_pen = QPen(QColor("black"), 1.4)
                self._scene.addEllipse(
                    marker_cx - MARKER_RADIUS,
                    marker_cy - MARKER_RADIUS,
                    MARKER_RADIUS * 2,
                    MARKER_RADIUS * 2,
                    circle_pen,
                    QBrush(fill_color),
                )

                min_x = min(min_x, marker_cx - MARKER_RADIUS)
                max_x = max(max_x, marker_cx + MARKER_RADIUS)
                min_y = min(min_y, marker_cy - MARKER_RADIUS, child_top)
                max_y = max(max_y, marker_cy + MARKER_RADIUS, child_top)

        # XOR / OR triangles
        for (parent, _group, kind), children in group_map.items():
            if not children:
                continue
            px, py = positions[parent]
            parent_bottom = (px, py + BOX_H / 2)
            child_positions = [positions[child] for child in children]
            child_positions_sorted = sorted(child_positions, key=lambda point: point[0])

            child_tops = [(child_x, child_y - BOX_H / 2) for child_x, child_y in child_positions_sorted]
            for child_x, child_top_y in child_tops:
                self._draw_line(parent_bottom[0], parent_bottom[1], child_x, child_top_y, width=1.4)

            min_child_top = min(child_top for _, child_top in child_tops)
            y_cross = parent_bottom[1] + max(TRIANGLE_GAP, (min_child_top - parent_bottom[1]) * 0.25)
            left_child = child_positions_sorted[0]
            right_child = child_positions_sorted[-1]

            def interp_x(x1: float, y1: float, x2: float, y2: float, y_target: float) -> float:
                t = (y_target - y1) / (y2 - y1) if y2 != y1 else 0.0
                return x1 + t * (x2 - x1)

            left_x = interp_x(
                parent_bottom[0],
                parent_bottom[1],
                left_child[0],
                left_child[1] - BOX_H / 2,
                y_cross,
            )
            right_x = interp_x(
                parent_bottom[0],
                parent_bottom[1],
                right_child[0],
                right_child[1] - BOX_H / 2,
                y_cross,
            )

            polygon = QPolygonF([
                QPointF(parent_bottom[0], parent_bottom[1]),
                QPointF(left_x, y_cross),
                QPointF(right_x, y_cross),
            ])
            fill_brush = QBrush(QColor("black") if kind == "or" else QColor("white"))
            pen = QPen(QColor("black"), 1.4)
            self._scene.addPolygon(polygon, pen, fill_brush)

            min_x = min(min_x, left_x, right_x, parent_bottom[0])
            max_x = max(max_x, left_x, right_x, parent_bottom[0])
            min_y = min(min_y, parent_bottom[1], y_cross)
            max_y = max(max_y, parent_bottom[1], y_cross)

        # Dependencies (dashed arrow)
        rects = {
            feature_id: (
                positions[feature_id][0] - width_map[feature_id] / 2,
                positions[feature_id][1] - BOX_H / 2,
                positions[feature_id][0] + width_map[feature_id] / 2,
                positions[feature_id][1] + BOX_H / 2,
            )
            for feature_id in positions
        }

        dep_lane_y = max(rect[3] for rect in rects.values()) + DEP_MARGIN
        used_lanes: List[float] = []
        structural_parents = {
            relation.parent for relation in relations if relation.kind in {"mandatory", "optional", "xor", "or"}
        }
        dep_index = 0

        for relation in relations:
            if relation.kind != "dependency":
                continue

            sx, sy = positions[relation.parent]
            tx, ty = positions[relation.child]
            sibling_like = abs(sy - ty) < BOX_H * 0.75
            adjacent_horizontal = sibling_like

            start_anchor, start_stub = anchor_with_stub(
                (sx, sy),
                (tx, ty),
                width_map[relation.parent],
                relation.parent not in structural_parents,
                force_bottom=False,
                force_side=adjacent_horizontal or False,
            )
            end_anchor, end_stub = anchor_with_stub(
                (tx, ty),
                (sx, sy),
                width_map[relation.child],
                relation.child not in structural_parents,
                force_bottom=False,
                force_side=adjacent_horizontal or False,
            )

            dash_offset = dep_index * 2
            dep_index += 1

            straight_start = start_anchor
            straight_end = end_anchor
            occludes = any(
                feature_id not in {relation.parent, relation.child}
                and segment_intersects_rect(straight_start, straight_end, rect)
                for feature_id, rect in rects.items()
            )

            if not occludes:
                points: List[Tuple[float, float]] = [start_anchor]
                if start_stub:
                    points.append(start_stub)
                if end_stub:
                    points.append(end_stub)
                points.append(end_anchor)
            else:
                start_anchor, start_stub = anchor_with_stub(
                    (sx, sy),
                    (tx, ty),
                    width_map[relation.parent],
                    relation.parent not in structural_parents,
                    force_bottom=relation.parent not in structural_parents,
                    force_side=relation.parent in structural_parents,
                )
                end_anchor, end_stub = anchor_with_stub(
                    (tx, ty),
                    (sx, sy),
                    width_map[relation.child],
                    relation.child not in structural_parents,
                    force_bottom=relation.child not in structural_parents,
                    force_side=relation.child in structural_parents,
                )

                lane_y = dep_lane_y + len(used_lanes) * DEP_LANE_SPACING
                used_lanes.append(lane_y)
                segments: List[Tuple[float, float]] = [start_anchor]
                if start_stub:
                    segments.append(start_stub)
                segments.append((segments[-1][0], lane_y))
                segments.append((end_anchor[0], lane_y))
                if end_stub:
                    segments.append(end_stub)
                segments.append(end_anchor)
                points = segments

            for point_a, point_b in zip(points, points[1:]):
                self._draw_line(
                    point_a[0],
                    point_a[1],
                    point_b[0],
                    point_b[1],
                    width=2.0,
                    color=QColor("#444444"),
                    dash=(float(DEP_DASH.split()[0]), float(DEP_DASH.split()[1])),
                    dash_offset=dash_offset,
                )
                min_x = min(min_x, point_a[0], point_b[0])
                max_x = max(max_x, point_a[0], point_b[0])
                min_y = min(min_y, point_a[1], point_b[1])
                max_y = max(max_y, point_a[1], point_b[1])

            if len(points) >= 2:
                point_a, point_b = points[-2], points[-1]
                angle = math.atan2(point_b[1] - point_a[1], point_b[0] - point_a[0])
                arrow_len = 10
                spread = math.pi / 8
                tip = point_b
                p1 = (
                    tip[0] - arrow_len * math.cos(angle - spread),
                    tip[1] - arrow_len * math.sin(angle - spread),
                )
                p2 = (
                    tip[0] - arrow_len * math.cos(angle + spread),
                    tip[1] - arrow_len * math.sin(angle + spread),
                )

                arrow_poly = QPolygonF([
                    QPointF(tip[0], tip[1]),
                    QPointF(p1[0], p1[1]),
                    QPointF(p2[0], p2[1]),
                ])
                dep_pen = QPen(QColor("#444444"), 1.2)
                self._scene.addPolygon(arrow_poly, dep_pen, QBrush(QColor("#444444")))

                min_x = min(min_x, tip[0], p1[0], p2[0])
                max_x = max(max_x, tip[0], p1[0], p2[0])
                min_y = min(min_y, tip[1], p1[1], p2[1])
                max_y = max(max_y, tip[1], p1[1], p2[1])

        width = max_x - min_x + 2 * PADDING
        height = max_y - min_y + 2 * PADDING
        self._scene.setSceneRect(min_x - PADDING, min_y - PADDING, width, height)

        self.resetTransform()
        self._zoom_level = 1.0
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _draw_centered_text(self, cx: float, cy: float, text: str) -> None:
        item = QGraphicsSimpleTextItem(text)
        item.setBrush(QBrush(QColor("black")))
        item.setFont(QFont("Arial", 10))
        self._scene.addItem(item)
        rect = item.boundingRect()
        item.setPos(cx - rect.width() / 2, cy - rect.height() / 2)

    def _draw_line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        width: float = 1.5,
        color: QColor = QColor("black"),
        dash: Optional[Tuple[float, float]] = None,
        dash_offset: float = 0.0,
    ) -> None:
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        if dash is not None:
            pen.setStyle(Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([dash[0], dash[1]])
            pen.setDashOffset(dash_offset)
        self._scene.addLine(x1, y1, x2, y2, pen)

    def _zoom_by_factor(self, factor: float) -> None:
        next_zoom = self._zoom_level * factor
        if next_zoom < 0.05 or next_zoom > 30.0:
            return
        self.scale(factor, factor)
        self._zoom_level = next_zoom

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._zoom_by_factor(1.14)
            event.accept()
            return
        if event.button() == Qt.MouseButton.RightButton:
            self._zoom_by_factor(1 / 1.14)
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_last_pos = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._panning and self._pan_last_pos is not None:
            delta = event.position() - self._pan_last_pos
            self._pan_last_pos = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._pan_last_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        modifiers = event.modifiers()
        if modifiers & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier):
            angle_delta = event.angleDelta().y()
            factor = 1.06 if angle_delta > 0 else 1 / 1.06
            self._zoom_by_factor(factor)
            event.accept()
            return
        super().wheelEvent(event)

    def event(self, event) -> bool:  # noqa: A003
        if event.type() == QEvent.Type.NativeGesture:
            native = event
            if isinstance(native, QNativeGestureEvent):
                if native.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
                    value = native.value()
                    factor = 1.0 + value
                    if factor > 0:
                        self._zoom_by_factor(factor)
                        return True
        return super().event(event)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self._scene.items() and self._zoom_level == 1.0:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
