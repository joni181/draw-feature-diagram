"""SVG renderer for feature diagrams."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

from .constants import (
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
from .geometry import anchor_with_stub, segment_intersects_rect
from .layout import LayoutResult
from .models import FeatureDiagram
from .svg_utils import svg_circle, svg_line, svg_polygon, svg_rect, svg_text


class SvgRenderer:
    """Render feature diagrams to SVG."""

    def render(self, diagram: FeatureDiagram, layout: LayoutResult, out_path: Path) -> None:
        features = diagram.features
        relations = diagram.relations
        positions = layout.positions
        width_map = layout.width_map
        group_map = layout.group_map

        shapes: List[str] = []
        min_x = min(cx - width_map[fid] / 2 for fid, (cx, _) in positions.items())
        max_x = max(cx + width_map[fid] / 2 for fid, (cx, _) in positions.items())
        min_y = min(cy for _, cy in positions.values()) - BOX_H / 2
        max_y = max(cy for _, cy in positions.values()) + BOX_H / 2

        # Boxes and labels
        for feat_id, feat in features.items():
            cx, cy = positions[feat_id]
            w = width_map[feat_id]
            rect_attrs = {"fill": "white", "stroke": "black", "stroke_width": "1.5"}
            if feat.is_reference:
                rect_attrs["stroke_dasharray"] = "6 4"
            shapes.append(svg_rect(cx, cy, w, BOX_H, **rect_attrs))
            shapes.append(
                svg_text(
                    cx,
                    cy,
                    feat.name,
                    font_size="14px",
                    fill="black",
                    font_family="Arial, sans-serif",
                )
            )

        # Mandatory / optional markers (with optional hub line for wide spans)
        mo_relations = [rel for rel in relations if rel.kind in {"mandatory", "optional"}]
        mo_children: Dict[str, List[str]] = {}
        for rel in mo_relations:
            mo_children.setdefault(rel.parent, []).append(rel.child)

        hub_y_map: Dict[str, float] = {}
        for parent, kids in mo_children.items():
            if len(kids) < 2:
                continue
            xs = [positions[ch][0] for ch in kids]
            span = max(xs) - min(xs)
            if span >= HUB_SPAN_THRESHOLD:
                py = positions[parent][1]
                parent_bottom = py + BOX_H / 2
                child_tops = [positions[ch][1] - BOX_H / 2 for ch in kids]
                avg_child_top = sum(child_tops) / len(child_tops)
                hub_y_map[parent] = parent_bottom + (avg_child_top - parent_bottom) * 0.5

        for parent, kids in mo_children.items():
            use_hub = parent in hub_y_map
            hub_y = hub_y_map.get(parent, 0.0)
            if use_hub:
                xs = [positions[ch][0] for ch in kids]
                line_start = (min(xs), hub_y)
                line_end = (max(xs), hub_y)
                shapes.append(
                    svg_line(
                        line_start[0],
                        line_start[1],
                        line_end[0],
                        line_end[1],
                        stroke="black",
                        stroke_width="1.4",
                    )
                )
                px, py = positions[parent]
                parent_bottom = (px, py + BOX_H / 2)
                shapes.append(
                    svg_line(
                        parent_bottom[0],
                        parent_bottom[1],
                        px,
                        hub_y,
                        stroke="black",
                        stroke_width="1.4",
                    )
                )
                min_x = min(min_x, line_start[0], line_end[0], parent_bottom[0])
                max_x = max(max_x, line_start[0], line_end[0], parent_bottom[0])
                min_y = min(min_y, hub_y, parent_bottom[1])
                max_y = max(max_y, hub_y, parent_bottom[1])

            for rel in mo_relations:
                if rel.parent != parent:
                    continue
                cx, cy = positions[rel.child]
                child_top = cy - BOX_H / 2
                marker_cy = child_top - MARKER_OFFSET
                marker_cx = cx
                if use_hub:
                    shapes.append(
                        svg_line(
                            marker_cx,
                            hub_y,
                            marker_cx,
                            marker_cy,
                            stroke="black",
                            stroke_width="1.4",
                        )
                    )
                else:
                    px, py = positions[parent]
                    start = (px, py + BOX_H / 2)
                    shapes.append(
                        svg_line(
                            start[0],
                            start[1],
                            marker_cx,
                            marker_cy,
                            stroke="black",
                            stroke_width="1.4",
                        )
                    )
                shapes.append(
                    svg_line(
                        marker_cx,
                        marker_cy,
                        cx,
                        child_top,
                        stroke="black",
                        stroke_width="1.4",
                    )
                )
                fill = "black" if rel.kind == "mandatory" else "white"
                shapes.append(
                    svg_circle(
                        marker_cx,
                        marker_cy,
                        MARKER_RADIUS,
                        stroke="black",
                        stroke_width="1.4",
                        fill=fill,
                    )
                )
                min_x = min(min_x, marker_cx) - MARKER_RADIUS
                max_x = max(max_x, marker_cx) + MARKER_RADIUS
                min_y = min(min_y, marker_cy, child_top, hub_y if use_hub else marker_cy) - MARKER_RADIUS
                max_y = max(max_y, marker_cy, child_top, hub_y if use_hub else marker_cy) + MARKER_RADIUS

        # XOR / OR triangles
        for (parent, _group, kind), children in group_map.items():
            if not children:
                continue
            px, py = positions[parent]
            parent_bottom = (px, py + BOX_H / 2)
            child_positions = [positions[ch] for ch in children]
            child_positions_sorted = sorted(child_positions, key=lambda c: c[0])

            # Draw connecting lines parent -> child tops
            child_tops = [(ch_cx, ch_cy - BOX_H / 2) for ch_cx, ch_cy in child_positions_sorted]

            for ch_cx, ch_top_y in child_tops:
                shapes.append(
                    svg_line(
                        parent_bottom[0],
                        parent_bottom[1],
                        ch_cx,
                        ch_top_y,
                        stroke="black",
                        stroke_width="1.4",
                    )
                )

            # Triangle constructed from a horizontal bar joining the connecting lines
            min_child_top = min(ch_top for _, ch_top in child_tops)
            y_cross = parent_bottom[1] + max(TRIANGLE_GAP, (min_child_top - parent_bottom[1]) * 0.25)
            left_child = child_positions_sorted[0]
            right_child = child_positions_sorted[-1]

            def interp_x(x1, y1, x2, y2, y_target) -> float:
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

            points = [parent_bottom, (left_x, y_cross), (right_x, y_cross)]
            fill = "black" if kind == "or" else "white"
            shapes.append(svg_polygon(points, fill=fill, stroke="black", stroke_width="1.4"))

            min_x = min(min_x, left_x, right_x, parent_bottom[0])
            max_x = max(max_x, left_x, right_x, parent_bottom[0])
            min_y = min(min_y, parent_bottom[1], y_cross)
            max_y = max(max_y, parent_bottom[1], y_cross)

        # Dependencies (dashed arrow)
        rects = {
            fid: (
                positions[fid][0] - width_map[fid] / 2,
                positions[fid][1] - BOX_H / 2,
                positions[fid][0] + width_map[fid] / 2,
                positions[fid][1] + BOX_H / 2,
            )
            for fid in positions
        }
        dep_lane_y = max(rect[3] for rect in rects.values()) + DEP_MARGIN
        used_lanes: List[float] = []
        structural_parents = {
            rel.parent for rel in relations if rel.kind in {"mandatory", "optional", "xor", "or"}
        }
        dep_index = 0

        for rel in relations:
            if rel.kind != "dependency":
                continue
            sx, sy = positions[rel.parent]
            tx, ty = positions[rel.child]
            sibling_like = abs(sy - ty) < BOX_H * 0.75
            adjacent_horizontal = sibling_like

            start_anchor, start_stub = anchor_with_stub(
                (sx, sy),
                (tx, ty),
                width_map[rel.parent],
                rel.parent not in structural_parents,
                force_bottom=False,
                force_side=adjacent_horizontal or False,
            )
            end_anchor, end_stub = anchor_with_stub(
                (tx, ty),
                (sx, sy),
                width_map[rel.child],
                rel.child not in structural_parents,
                force_bottom=False,
                force_side=adjacent_horizontal or False,
            )
            dep_color = "#444444"
            dash_offset = dep_index * 2
            dep_index += 1

            # Determine if straight line between anchors would hit any other box.
            straight_start = start_anchor
            straight_end = end_anchor
            occludes = any(
                fid not in {rel.parent, rel.child}
                and segment_intersects_rect(straight_start, straight_end, rect)
                for fid, rect in rects.items()
            )

            if not occludes:
                points: List[Tuple[float, float]] = [start_anchor]
                if start_stub:
                    points.append(start_stub)
                if end_stub:
                    points.append(end_stub)
                points.append(end_anchor)
            else:
                # For routed lines, prefer bottom entry for leaves, side for others.
                start_anchor, start_stub = anchor_with_stub(
                    (sx, sy),
                    (tx, ty),
                    width_map[rel.parent],
                    rel.parent not in structural_parents,
                    force_bottom=rel.parent not in structural_parents,
                    force_side=rel.parent in structural_parents,
                )
                end_anchor, end_stub = anchor_with_stub(
                    (tx, ty),
                    (sx, sy),
                    width_map[rel.child],
                    rel.child not in structural_parents,
                    force_bottom=rel.child not in structural_parents,
                    force_side=rel.child in structural_parents,
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

            # Draw segments
            for a, b in zip(points, points[1:]):
                shapes.append(
                    svg_line(
                        a[0],
                        a[1],
                        b[0],
                        b[1],
                        stroke=dep_color,
                        stroke_width="2",
                        stroke_dasharray=DEP_DASH,
                        stroke_dashoffset=str(dash_offset),
                        stroke_linecap="round",
                    )
                )
                min_x = min(min_x, a[0], b[0])
                max_x = max(max_x, a[0], b[0])
                min_y = min(min_y, a[1], b[1])
                max_y = max(max_y, a[1], b[1])

            # Arrowhead at final segment
            if len(points) >= 2:
                a, b = points[-2], points[-1]
                angle = math.atan2(b[1] - a[1], b[0] - a[0])
                arrow_len = 10
                spread = math.pi / 8
                tip = b
                p1 = (
                    tip[0] - arrow_len * math.cos(angle - spread),
                    tip[1] - arrow_len * math.sin(angle - spread),
                )
                p2 = (
                    tip[0] - arrow_len * math.cos(angle + spread),
                    tip[1] - arrow_len * math.sin(angle + spread),
                )
                shapes.append(
                    svg_polygon(
                        [tip, p1, p2],
                        fill=dep_color,
                        stroke=dep_color,
                        stroke_width="1.2",
                    )
                )
                min_x = min(min_x, tip[0], p1[0], p2[0])
                max_x = max(max_x, tip[0], p1[0], p2[0])
                min_y = min(min_y, tip[1], p1[1], p2[1])
                max_y = max(max_y, tip[1], p1[1], p2[1])

        width = max_x - min_x + 2 * PADDING
        height = max_y - min_y + 2 * PADDING
        offset_x = -min_x + PADDING
        offset_y = -min_y + PADDING

        content = "\n  ".join(shapes)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.1f}" height="{height:.1f}" '
            f'viewBox="0 0 {width:.1f} {height:.1f}">\n'
            f'  <g transform="translate({offset_x:.1f},{offset_y:.1f})">\n'
            f'  {content}\n'
            f'  </g>\n'
            f'</svg>\n'
        )
        out_path.write_text(svg, encoding="utf-8")

