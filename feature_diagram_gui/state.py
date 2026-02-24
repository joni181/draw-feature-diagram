"""Document state and persistence helpers for the feature diagram GUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from feature_diagram_core.layout import FeatureDiagramLayoutEngine
from feature_diagram_core.models import Feature, FeatureDiagram, ModelParseError, Relation
from feature_diagram_core.renderer import SvgRenderer


RELATION_KINDS = ("mandatory", "optional", "or", "xor", "dependency")


@dataclass
class FeatureItem:
    feature_id: str
    name: str
    is_reference: bool = False


@dataclass
class RelationItem:
    kind: str
    parent: str
    child: str
    group: str = ""


class DiagramDocument:
    """In-memory editable diagram document."""

    def __init__(self) -> None:
        self.features: List[FeatureItem] = []
        self.relations: List[RelationItem] = []
        self.json_path: Optional[Path] = None
        self.export_path: Optional[Path] = None
        self.dirty = False

    @property
    def feature_ids(self) -> List[str]:
        return [feature.feature_id for feature in self.features]

    def feature_index_by_id(self, feature_id: str) -> Optional[int]:
        for idx, feature in enumerate(self.features):
            if feature.feature_id == feature_id:
                return idx
        return None

    def set_dirty(self, value: bool = True) -> None:
        self.dirty = value

    def clear(self) -> None:
        self.features.clear()
        self.relations.clear()
        self.json_path = None
        self.export_path = None
        self.dirty = False

    def add_feature(self, feature: FeatureItem) -> None:
        self.features.append(feature)
        self.set_dirty()

    def update_feature(self, index: int, updated: FeatureItem) -> None:
        old_id = self.features[index].feature_id
        new_id = updated.feature_id
        self.features[index] = updated
        if old_id != new_id:
            for relation in self.relations:
                if relation.parent == old_id:
                    relation.parent = new_id
                if relation.child == old_id:
                    relation.child = new_id
                if relation.group == old_id:
                    relation.group = new_id
        self.set_dirty()

    def delete_feature_at(self, index: int) -> None:
        feature_id = self.features[index].feature_id
        del self.features[index]
        self.relations = [
            relation
            for relation in self.relations
            if relation.parent != feature_id and relation.child != feature_id
        ]
        self.set_dirty()

    def reorder_feature(self, from_index: int, to_index: int) -> None:
        if from_index == to_index:
            return
        feature = self.features.pop(from_index)
        self.features.insert(to_index, feature)
        self.set_dirty()

    def add_relation(self, relation: RelationItem) -> None:
        self.relations.append(relation)
        self.set_dirty()

    def update_relation(self, index: int, relation: RelationItem) -> None:
        self.relations[index] = relation
        self.set_dirty()

    def delete_relation_at(self, index: int) -> None:
        del self.relations[index]
        self.set_dirty()

    def reorder_relation(self, from_index: int, to_index: int) -> None:
        if from_index == to_index:
            return
        relation = self.relations.pop(from_index)
        self.relations.insert(to_index, relation)
        self.set_dirty()

    def to_json_payload(self) -> Dict[str, object]:
        return {
            "features": [
                {"id": feature.feature_id, "name": feature.name}
                for feature in self.features
                if not feature.is_reference
            ],
            "reference_features": [
                {"id": feature.feature_id, "name": feature.name}
                for feature in self.features
                if feature.is_reference
            ],
            "relations": [
                {
                    "kind": relation.kind,
                    "parent": relation.parent,
                    "child": relation.child,
                    **({"group": relation.group} if relation.group else {}),
                }
                for relation in self.relations
            ],
        }

    def from_json_payload(self, payload: Dict[str, object]) -> None:
        if "relations" not in payload:
            raise ModelParseError("JSON must contain a 'relations' field.")
        if "features" not in payload and "reference_features" not in payload:
            raise ModelParseError(
                "JSON must contain either 'features' or 'reference_features' (and 'relations')."
            )

        loaded_features: List[FeatureItem] = []
        feature_ids = set()

        for feature in payload.get("features", []):
            feature_id = str(feature["id"])
            if feature_id in feature_ids:
                raise ModelParseError(f"Duplicate feature id '{feature_id}'.")
            loaded_features.append(
                FeatureItem(
                    feature_id=feature_id,
                    name=str(feature.get("name", feature_id)),
                    is_reference=False,
                )
            )
            feature_ids.add(feature_id)

        for feature in payload.get("reference_features", []):
            feature_id = str(feature["id"])
            if feature_id in feature_ids:
                raise ModelParseError(f"Duplicate feature id '{feature_id}'.")
            loaded_features.append(
                FeatureItem(
                    feature_id=feature_id,
                    name=str(feature.get("name", feature_id)),
                    is_reference=True,
                )
            )
            feature_ids.add(feature_id)

        loaded_relations: List[RelationItem] = []
        for relation in payload["relations"]:
            kind = str(relation["kind"]).lower()
            if kind not in RELATION_KINDS:
                raise ModelParseError(f"Unsupported relation '{kind}'.")
            parent = str(relation["parent"])
            child = str(relation["child"])
            group = str(relation.get("group", ""))
            loaded_relations.append(RelationItem(kind=kind, parent=parent, child=child, group=group))
            if parent not in feature_ids:
                loaded_features.append(FeatureItem(feature_id=parent, name=parent))
                feature_ids.add(parent)
            if child not in feature_ids:
                loaded_features.append(FeatureItem(feature_id=child, name=child))
                feature_ids.add(child)

        self.features = loaded_features
        self.relations = loaded_relations
        self.set_dirty(False)

    def load_json_file(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.from_json_payload(payload)
        self.json_path = path
        self.export_path = None

    def save_json_file(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_json_payload(), indent=2), encoding="utf-8")
        self.json_path = path
        self.set_dirty(False)

    def to_core_diagram(self) -> FeatureDiagram:
        features = {
            feature.feature_id: Feature(
                feature_id=feature.feature_id,
                name=feature.name,
                is_reference=feature.is_reference,
            )
            for feature in self.features
        }
        relations = [
            Relation(
                kind=relation.kind,
                parent=relation.parent,
                child=relation.child,
                group=relation.group or None,
            )
            for relation in self.relations
        ]
        return FeatureDiagram(features=features, relations=relations)

    def export_svg(self, path: Path) -> None:
        diagram = self.to_core_diagram()
        layout = FeatureDiagramLayoutEngine().compute(diagram)
        SvgRenderer().render(diagram, layout, path)
        self.export_path = path


def relation_to_display(relation: RelationItem) -> str:
    group = relation.group if relation.group else "-"
    return f"{relation.kind}  {relation.parent} -> {relation.child}  [{group}]"
