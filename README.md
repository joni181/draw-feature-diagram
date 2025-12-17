# Feature Diagram Renderer

Command-line tool that turns a JSON feature model into a feature diagram SVG. No Graphviz is used; all shapes are drawn directly in SVG. Boxes show feature names and auto-size to fit labels (with a minimum width), circles on the incoming relation encode mandatory/optional, triangles under a parent encode XOR/OR groups, and dashed arrows encode dependencies.

## JSON model format
- Root object fields:
  - `features`: array of objects with `id` (string, unique) and `name` (string, label displayed in the box).
  - `relations`: array of objects describing edges:
    - `kind`: one of `mandatory`, `optional`, `xor`, `or`, `dependency`.
    - `parent`: feature id of the source.
    - `child`: feature id of the target.
    - `group` (optional): group identifier used to cluster XOR/OR children that belong to the same alternative set. If omitted, the parent id is used.
- Semantics:
  - `mandatory` / `optional`: connects parent → circle → child (filled for mandatory, empty for optional) with straight lines.
  - `xor`: children are mutually exclusive; rendered via an empty triangle under the parent, with lines to children.
  - `or`: at least one child; rendered via a filled triangle under the parent, with lines to children.
  - `dependency`: dashed arrow `parent - - -> child`.

### Minimal example
```json
{
  "features": [
    { "id": "Root", "name": "Root Feature" },
    { "id": "A", "name": "Feature A" },
    { "id": "B", "name": "Feature B" },
    { "id": "C1", "name": "Choice 1" },
    { "id": "C2", "name": "Choice 2" }
  ],
  "relations": [
    { "kind": "mandatory", "parent": "Root", "child": "A" },
    { "kind": "optional", "parent": "Root", "child": "B" },
    { "kind": "xor", "parent": "Root", "child": "C1", "group": "choices" },
    { "kind": "xor", "parent": "Root", "child": "C2", "group": "choices" },
    { "kind": "dependency", "parent": "A", "child": "B" }
  ]
}
```

## Usage
Run directly (SVG output):
```bash
source .venv/bin/activate
python feature_diagram.py model.json --out diagram.svg
```

Flags:
- `json_file`: path to the input model.
- `--out`: output SVG path (default `feature-diagram.svg`).
- `--write-json`: echo the parsed model to a new JSON file (useful after auto-filling missing features).
