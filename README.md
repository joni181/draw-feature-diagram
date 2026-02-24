# Feature Diagram Renderer

Command-line tool that turns a JSON feature model into a feature diagram SVG. No Graphviz is used; all shapes are drawn directly in SVG. Boxes show feature names and auto-size to fit labels (with a minimum width), circles on the incoming relation encode mandatory/optional, triangles under a parent encode XOR/OR groups, dashed arrows encode dependencies, and reference features are rendered as dashed-outline boxes.

## JSON model format
- Root object fields:
  - `features`: array of objects with `id` (string, unique) and `name` (string, label displayed in the box).
  - `reference_features` (optional): array of objects with `id` and `name`, same schema as `features`, rendered as dashed-outline boxes to indicate referenced/off-page features.
  - `relations`: array of objects describing edges:
    - `kind`: one of `mandatory`, `optional`, `xor`, `or`, `dependency`.
    - `parent`: feature id of the source.
    - `child`: feature id of the target.
    - `group` (optional): group identifier used to cluster XOR/OR children that belong to the same alternative set. If omitted, the parent id is used.
- Semantics:
  - `reference_features`: rendered identically to normal features, except their box outline is dashed.
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
  "reference_features": [
    { "id": "ExternalX", "name": "External Subtree X" }
  ],
  "relations": [
    { "kind": "mandatory", "parent": "Root", "child": "A" },
    { "kind": "optional", "parent": "Root", "child": "B" },
    { "kind": "xor", "parent": "Root", "child": "C1", "group": "choices" },
    { "kind": "xor", "parent": "Root", "child": "C2", "group": "choices" },
    { "kind": "dependency", "parent": "A", "child": "B" },
    { "kind": "dependency", "parent": "B", "child": "ExternalX" }
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
- `--write-json`: echo the parsed model to a new JSON file (including both `features` and `reference_features`).

## GUI editor (live preview)
Launch the desktop editor:
```bash
source .venv/bin/activate
pip install -r requirements.txt
python feature_diagram_editor.py
```

Editor highlights:
- Two-column layout: editor pane (features + relations lists) and live preview pane.
- Feature list supports normal and reference features in one list.
- Add feature dialog:
  - asks for `name`
  - auto-suggests `id` as lowercased name with underscores and stripped special chars
  - resolves collisions as `feature_id-1`, `feature_id-2`, ...
  - allows manual `id` edits
  - includes a reference-feature toggle
- Feature and relation edit mode:
  - drag by handle (`≡`) to reorder
  - delete by trash icon (`🗑`) with no confirmation
- Feature and relation list headers include icon buttons for add (`+`), remove selected (`−`), and edit mode (`✎` / `✓`).
- Click list rows (outside edit mode) to edit details inline in the editor pane (no dialog), with `Apply`, `Delete`, and close (`X`).
- Load / Save / Save As / Export / Re-export controls in the top toolbar.
- Preview updates automatically after add/edit/delete/reorder.
- Preview navigation:
  - left click: zoom in
  - right click: zoom out
  - middle button drag: pan
  - wheel/trackpad scroll: pan
  - ctrl/cmd + wheel: zoom
  - trackpad pinch (macOS native gesture): zoom

Note: the GUI uses PySide6.
