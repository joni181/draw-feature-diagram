"""Small SVG helper builders."""

from __future__ import annotations

from typing import List, Tuple
from xml.sax.saxutils import escape


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

