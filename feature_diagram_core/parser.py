"""JSON parsing for feature diagram models."""

from __future__ import annotations

import json
from pathlib import Path

from .models import Feature, FeatureDiagram, JsonFeature, JsonModel, ModelParseError, Relation


def parse_json_model(path: Path) -> FeatureDiagram:
    """Parse a JSON file into a feature diagram model."""
    try:
        payload: JsonModel = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelParseError(f"Invalid JSON: {exc}") from exc
    if "relations" not in payload:
        raise ModelParseError("JSON must contain a 'relations' field.")
    if "features" not in payload and "reference_features" not in payload:
        raise ModelParseError(
            "JSON must contain either 'features' or 'reference_features' (and 'relations')."
        )

    features = {}
    relations = []

    def add_feature(feat: JsonFeature, is_reference: bool) -> None:
        fid = feat["id"]
        features[fid] = Feature(
            feature_id=fid,
            name=feat.get("name", fid),
            is_reference=is_reference,
        )

    for feat in payload.get("features", []):
        add_feature(feat, is_reference=False)

    for feat in payload.get("reference_features", []):
        add_feature(feat, is_reference=True)

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

    return FeatureDiagram(features=features, relations=relations)

