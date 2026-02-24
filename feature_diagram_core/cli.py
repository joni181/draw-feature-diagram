"""CLI entrypoint for the feature diagram renderer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .layout import FeatureDiagramLayoutEngine
from .models import ModelParseError
from .parser import parse_json_model
from .renderer import SvgRenderer


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
        diagram = parse_json_model(args.json_file)
    except (OSError, ModelParseError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    layout = FeatureDiagramLayoutEngine().compute(diagram)

    if args.write_json:
        args.write_json.write_text(
            json.dumps(diagram.to_json_payload(), indent=2),
            encoding="utf-8",
        )

    SvgRenderer().render(diagram, layout, args.out)
    return 0

