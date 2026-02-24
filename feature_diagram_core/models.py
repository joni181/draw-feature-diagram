"""Data model definitions for feature diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict


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
    reference_features: List[JsonFeature]
    relations: List[JsonRelation]


@dataclass
class Feature:
    feature_id: str
    name: str
    is_reference: bool = False


@dataclass
class Relation:
    kind: RelationKind
    parent: str
    child: str
    group: Optional[str] = None


@dataclass
class FeatureDiagram:
    features: Dict[str, Feature]
    relations: List[Relation]

    def to_json_payload(self) -> JsonModel:
        return {
            "features": [
                {"id": feature.feature_id, "name": feature.name}
                for feature in self.features.values()
                if not feature.is_reference
            ],
            "reference_features": [
                {"id": feature.feature_id, "name": feature.name}
                for feature in self.features.values()
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


class ModelParseError(Exception):
    pass

