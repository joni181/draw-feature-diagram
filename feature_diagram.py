#!/usr/bin/env python3
"""
Render a feature diagram from a structured JSON description without using Graphviz.

The script draws an SVG with boxes for features, circles on mandatory/optional
edges, triangles for XOR/OR groups, and dashed arrows for dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict
from xml.sax.saxutils import escape


RelationKind = str


class JsonRelation(TypedDict, total=False):
    kind: str
    parent: str
    child: str
    group: str


class JsonFeature(TypedDict):
    id: str
    name: str


class JsonModel(TypedDict, total=False):
    features: List[JsonFeature]
    relations: List[JsonRelation]


@dataclass
class Feature:
    feature_id: str
    name: str


@dataclass
class Relation:
    kind: RelationKind
    parent: str
    child: str
    group: Optional[str] = None


class ModelParseError(Exception):
    pass


def parse_json_model(path: Path) -> Tuple[Dict[str, Feature], List[Relation]]:
    """Parse a JSON file into features and relations."""
    try:
        payload: JsonModel = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelParseError(f"Invalid JSON: {exc}") from exc
    if "features" not in payload or "relations" not in payload:
        raise ModelParseError("JSON must contain 'features' and 'relations' fields.")

    features: Dict[str, Feature] = {}
    relations: List[Relation] = []

    for feat in payload["features"]:
        fid = feat["id"]
        features[fid] = Feature(feature_id=fid, name=feat.get("name", fid))

    for rel in payload["relations"]:
        kind = rel["kind"].lower()
        if kind not in {"mandatory", "optional", "xor", "or", "dependency"}:
            raise ModelParseError(f"Unsupported relation '{kind}'.")
        parent = rel["parent"]
        child = rel["child"]
        group = rel.get("group")
        relations.append(Relation(kind=kind, parent=parent, child=child, group=group))
        if parent not in features:
            features[parent] = Feature(feature_id=parent, name=parent)
        if child not in features:
            features[child] = Feature(feature_id=child, name=child)

    if not features:
        raise ModelParseError("No features found in the JSON file.")

    return features, relations


# Layout constants
MIN_BOX_W = 90
BOX_H = 45
H_SPACING = 60
V_SPACING = 140
PADDING = 40
MARKER_OFFSET = 8
MARKER_RADIUS = 6
TRIANGLE_H = 16
TRIANGLE_GAP = 8
TEXT_PADDING_X = 16
AVG_CHAR_W = 7
DEP_LANE_SPACING = 24
DEP_MARGIN = 30
DEP_ORTHO_OFFSET = 12
DEP_DASH = "5 4"


def build_child_map(relations: List[Relation]) -> Dict[str, List[str]]:
    """Map parent -> ordered children for tree-like relations."""
    child_map: Dict[str, List[str]] = {}
    for rel in relations:
        if rel.kind in {"mandatory", "optional", "xor", "or"}:
            child_map.setdefault(rel.parent, []).append(rel.child)
    return child_map


def build_group_map(relations: List[Relation]) -> Dict[Tuple[str, str, str], List[str]]:
    """Map (parent, group, kind) -> children for XOR/OR groups."""
    group_relations: Dict[Tuple[str, str, str], List[str]] = {}
    for rel in relations:
        if rel.kind in {"xor", "or"}:
            key = (rel.parent, rel.group or rel.parent, rel.kind)
            group_relations.setdefault(key, []).append(rel.child)
    return group_relations


def compute_roots(features: Dict[str, Feature], relations: List[Relation]) -> List[str]:
    children = {
        rel.child
        for rel in relations
        if rel.kind in {"mandatory", "optional", "xor", "or"}
    }
    roots = [fid for fid in features if fid not in children]
    if not roots:
        # fallback: pick arbitrary to avoid crash
        roots = [next(iter(features))]
    return roots


def compute_widths(features: Dict[str, Feature]) -> Dict[str, float]:
    widths: Dict[str, float] = {}
    for fid, feat in features.items():
        estimated = AVG_CHAR_W * len(feat.name) + 2 * TEXT_PADDING_X
        widths[fid] = max(MIN_BOX_W, estimated)
    return widths


def compute_subtree_width(
    node: str,
    child_map: Dict[str, List[str]],
    width_map: Dict[str, float],
    cache: Dict[str, float],
) -> float:
    if node in cache:
        return cache[node]
    children = child_map.get(node, [])
    if not children:
        cache[node] = width_map[node]
        return width_map[node]
    widths = [compute_subtree_width(ch, child_map, width_map, cache) for ch in children]
    total = sum(widths) + H_SPACING * (len(children) - 1)
    cache[node] = max(width_map[node], total)
    return cache[node]


def assign_positions(
    node: str,
    x_left: float,
    y: float,
    child_map: Dict[str, List[str]],
    width_map: Dict[str, float],
    width_cache: Dict[str, float],
    positions: Dict[str, Tuple[float, float]],
) -> float:
    """Assign x/y centers recursively. Returns subtree width placed."""
    node_width = width_cache[node]
    children = child_map.get(node, [])
    if not children:
        positions[node] = (x_left + node_width / 2, y)
        return node_width

    child_widths = [width_cache[ch] for ch in children]
    child_total = sum(child_widths) + H_SPACING * (len(children) - 1)
    offset_start = (max(node_width, child_total) - child_total) / 2
    child_x = x_left + offset_start
    for ch, w in zip(children, child_widths):
        assign_positions(ch, child_x, y + V_SPACING, child_map, width_map, width_cache, positions)
        child_x += w + H_SPACING

    positions[node] = (x_left + max(node_width, child_total) / 2, y)
    return max(node_width, child_total)


def svg_line(x1, y1, x2, y2, **attrs) -> str:
    attr_str = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" {attr_str}/>'


def svg_rect(cx, cy, w, h, **attrs) -> str:
    x = cx - w / 2
    y = cy - h / 2
    attr_str = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" {attr_str}/>'


def svg_circle(cx, cy, r, **attrs) -> str:
    attr_str = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" {attr_str}/>'


def svg_polygon(points: List[Tuple[float, float]], **attrs) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    attr_str = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<polygon points="{pts}" {attr_str}/>'


def svg_text(cx, cy, text, **attrs) -> str:
    attr_str = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    safe = escape(text)
    return f'<text x="{cx:.1f}" y="{cy:.1f}" text-anchor="middle" dominant-baseline="middle" {attr_str}>{safe}</text>'


def rect_edge_point(
    cx: float, cy: float, tx: float, ty: float, w: float, h: float = BOX_H
) -> Tuple[float, float]:
    """Point on rectangle boundary from center toward target."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    half_w, half_h = w / 2, h / 2
    scale = 1.0 / max(abs(dx) / half_w if half_w else 1, abs(dy) / half_h if half_h else 1)
    return cx + dx * scale, cy + dy * scale


def point_in_rect(x: float, y: float, rect: Tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = rect
    return minx <= x <= maxx and miny <= y <= maxy


def line_intersect(p1, p2, q1, q2) -> bool:
    (x1, y1), (x2, y2) = p1, p2
    (x3, y3), (x4, y4) = q1, q2

    def orient(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    o1 = orient(x1, y1, x2, y2, x3, y3)
    o2 = orient(x1, y1, x2, y2, x4, y4)
    o3 = orient(x3, y3, x4, y4, x1, y1)
    o4 = orient(x3, y3, x4, y4, x2, y2)

    def on_segment(ax, ay, bx, by, cx, cy):
        return min(ax, bx) <= cx <= max(ax, bx) and min(ay, by) <= cy <= max(ay, by)

    if o1 == 0 and on_segment(x1, y1, x2, y2, x3, y3):
        return True
    if o2 == 0 and on_segment(x1, y1, x2, y2, x4, y4):
        return True
    if o3 == 0 and on_segment(x3, y3, x4, y4, x1, y1):
        return True
    if o4 == 0 and on_segment(x3, y3, x4, y4, x2, y2):
        return True

    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)


def segment_intersects_rect(p1, p2, rect: Tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = rect
    if point_in_rect(p1[0], p1[1], rect) or point_in_rect(p2[0], p2[1], rect):
        return True
    corners = [(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return any(line_intersect(p1, p2, a, b) for a, b in edges)


def anchor_with_stub(
    src: Tuple[float, float],
    dst: Tuple[float, float],
    src_w: float,
    leaf: bool,
) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]]]:
    """Compute anchor point on source box and optional stub for orthogonal exit."""
    cx, cy = src
    tx, ty = dst
    if leaf:
        anchor = (cx, cy + BOX_H / 2)
        return anchor, None
    dx = tx - cx
    side = 1 if dx >= 0 else -1
    anchor = (cx + side * src_w / 2, cy)
    stub = (anchor[0] + side * DEP_ORTHO_OFFSET, anchor[1])
    return anchor, stub


def render_svg(
    features: Dict[str, Feature],
    relations: List[Relation],
    positions: Dict[str, Tuple[float, float]],
    width_map: Dict[str, float],
    child_map: Dict[str, List[str]],
    group_map: Dict[Tuple[str, str, str], List[str]],
    out_path: Path,
) -> None:
    shapes: List[str] = []
    min_x = min(cx - width_map[fid] / 2 for fid, (cx, _) in positions.items())
    max_x = max(cx + width_map[fid] / 2 for fid, (cx, _) in positions.items())
    min_y = min(cy for _, cy in positions.values()) - BOX_H / 2
    max_y = max(cy for _, cy in positions.values()) + BOX_H / 2

    # Boxes and labels
    for feat_id, feat in features.items():
        cx, cy = positions[feat_id]
        w = width_map[feat_id]
        shapes.append(svg_rect(cx, cy, w, BOX_H, fill="white", stroke="black", stroke_width="1.5"))
        shapes.append(svg_text(cx, cy, feat.name, font_size="14px", fill="black", font_family="Arial, sans-serif"))

    # Mandatory / optional markers
    for rel in relations:
        if rel.kind not in {"mandatory", "optional"}:
            continue
        px, py = positions[rel.parent]
        cx, cy = positions[rel.child]
        start = (px, py + BOX_H / 2)
        end = (cx, cy - BOX_H / 2)
        marker_cy = end[1] - MARKER_OFFSET
        marker_cx = end[0]
        shapes.append(svg_line(start[0], start[1], marker_cx, marker_cy, stroke="black", stroke_width="1.4"))
        shapes.append(svg_line(marker_cx, marker_cy, end[0], end[1], stroke="black", stroke_width="1.4"))
        fill = "black" if rel.kind == "mandatory" else "white"
        shapes.append(svg_circle(marker_cx, marker_cy, MARKER_RADIUS, stroke="black", stroke_width="1.4", fill=fill))
        min_x = min(min_x, start[0], end[0], marker_cx) - MARKER_RADIUS
        max_x = max(max_x, start[0], end[0], marker_cx) + MARKER_RADIUS
        min_y = min(min_y, start[1], end[1], marker_cy) - MARKER_RADIUS
        max_y = max(max_y, start[1], end[1], marker_cy) + MARKER_RADIUS

    # XOR / OR triangles
    for (parent, _group, kind), children in group_map.items():
        if not children:
            continue
        px, py = positions[parent]
        parent_w = width_map[parent]
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

        left_x = interp_x(parent_bottom[0], parent_bottom[1], left_child[0], left_child[1] - BOX_H / 2, y_cross)
        right_x = interp_x(parent_bottom[0], parent_bottom[1], right_child[0], right_child[1] - BOX_H / 2, y_cross)

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
            rel.parent not in structural_parents and not adjacent_horizontal,
        )
        end_anchor, end_stub = anchor_with_stub(
            (tx, ty),
            (sx, sy),
            width_map[rel.child],
            rel.child not in structural_parents and not adjacent_horizontal,
        )
        dep_color = "#444444"
        dash_offset = dep_index * 2
        dep_index += 1

        # Determine if straight line between anchors would hit any other box.
        straight_start = start_anchor
        straight_end = end_anchor
        occludes = any(
            fid not in {rel.parent, rel.child} and segment_intersects_rect(straight_start, straight_end, rect)
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
            shapes.append(svg_polygon([tip, p1, p2], fill=dep_color, stroke=dep_color, stroke_width="1.2"))
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a feature diagram with XOR, OR, mandatory/optional markers, and dependencies (SVG, no Graphviz)."
    )
    parser.add_argument("json_file", type=Path, help="Input JSON file describing the feature model.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("feature-diagram.svg"),
        help="Output SVG path.",
    )
    parser.add_argument(
        "--write-json",
        type=Path,
        help="Optional path to write the parsed model as structured JSON (echo).",
    )

    args = parser.parse_args(argv)

    try:
        features, relations = parse_json_model(args.json_file)
    except (OSError, ModelParseError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    child_map = build_child_map(relations)
    group_map = build_group_map(relations)
    roots = compute_roots(features, relations)
    width_map = compute_widths(features)

    width_cache: Dict[str, float] = {}
    positions: Dict[str, Tuple[float, float]] = {}
    current_x = 0.0
    for root in roots:
        w = compute_subtree_width(root, child_map, width_map, width_cache)
        assign_positions(root, current_x, 0.0, child_map, width_map, width_cache, positions)
        current_x += w + H_SPACING

    if args.write_json:
        args.write_json.write_text(
            json.dumps(
                {
                    "features": [{"id": f.feature_id, "name": f.name} for f in features.values()],
                    "relations": [
                        {
                            "kind": r.kind,
                            "parent": r.parent,
                            "child": r.child,
                            **({"group": r.group} if r.group else {}),
                        }
                        for r in relations
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    render_svg(features, relations, positions, width_map, child_map, group_map, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
