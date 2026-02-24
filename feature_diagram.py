#!/usr/bin/env python3
"""Render feature diagrams from JSON to SVG."""

from __future__ import annotations

import sys

from feature_diagram_core.cli import main


if __name__ == "__main__":
    sys.exit(main())
