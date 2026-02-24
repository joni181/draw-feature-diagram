"""Layout engine for feature diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .constants import AVG_CHAR_W, H_SPACING, MIN_BOX_W, TEXT_PADDING_X, V_SPACING
from .models import FeatureDiagram, Relation


@dataclass
class LayoutResult:
    positions: Dict[str, Tuple[float, float]]
    width_map: Dict[str, float]
    child_map: Dict[str, List[str]]
    group_map: Dict[Tuple[str, str, str], List[str]]


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


def compute_roots(features: Dict[str, object], relations: List[Relation]) -> List[str]:
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


def compute_widths(diagram: FeatureDiagram) -> Dict[str, float]:
    widths: Dict[str, float] = {}
    for fid, feat in diagram.features.items():
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


class FeatureDiagramLayoutEngine:
    """Compute diagram layout while preserving the original algorithm."""

    def compute(self, diagram: FeatureDiagram) -> LayoutResult:
        child_map = build_child_map(diagram.relations)
        group_map = build_group_map(diagram.relations)
        roots = compute_roots(diagram.features, diagram.relations)
        width_map = compute_widths(diagram)

        width_cache: Dict[str, float] = {}
        positions: Dict[str, Tuple[float, float]] = {}
        current_x = 0.0
        for root in roots:
            w = compute_subtree_width(root, child_map, width_map, width_cache)
            assign_positions(root, current_x, 0.0, child_map, width_map, width_cache, positions)
            current_x += w + H_SPACING

        return LayoutResult(
            positions=positions,
            width_map=width_map,
            child_map=child_map,
            group_map=group_map,
        )

