"""Geometry helpers used by diagram rendering."""

from __future__ import annotations

from typing import Optional, Tuple

from .constants import BOX_H, DEP_ORTHO_OFFSET


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
    force_bottom: bool = False,
    force_side: bool = False,
) -> Tuple[Tuple[float, float], Optional[Tuple[float, float]]]:
    """Compute anchor point on source box and optional stub for orthogonal exit."""
    cx, cy = src
    tx, ty = dst
    dx = tx - cx
    dy = ty - cy

    # Prefer vertical anchors if the movement is mostly vertical
    if force_bottom:
        anchor = (cx, cy + BOX_H / 2)
        if leaf:
            return anchor, None
        return anchor, (anchor[0], anchor[1] + DEP_ORTHO_OFFSET)

    if force_side:
        side = 1 if dx >= 0 else -1
        anchor = (cx + side * src_w / 2, cy)
        if leaf:
            return anchor, None
        return anchor, (anchor[0] + side * DEP_ORTHO_OFFSET, anchor[1])

    if abs(dy) >= abs(dx):
        if leaf:
            anchor = (cx, cy - BOX_H / 2) if dy < 0 else (cx, cy + BOX_H / 2)
            return anchor, None
        anchor = (cx, cy - BOX_H / 2) if dy < 0 else (cx, cy + BOX_H / 2)
        stub = (anchor[0], anchor[1] - DEP_ORTHO_OFFSET if dy < 0 else anchor[1] + DEP_ORTHO_OFFSET)
        return anchor, stub

    # Horizontal preference
    side = 1 if dx >= 0 else -1
    if leaf:
        anchor = (cx + side * src_w / 2, cy)
        return anchor, None
    anchor = (cx + side * src_w / 2, cy)
    stub = (anchor[0] + side * DEP_ORTHO_OFFSET, anchor[1])
    return anchor, stub

