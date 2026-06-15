"""Strip notebook cells to headings + code only, for follow-along use.

Markdown cells are reduced to heading lines only. Headings that appear
inside an `#exm-` div block in the source qmd are prefixed with "Example: "
so students can track which example each code block belongs to.
"""
import json
import re
from pathlib import Path


def extract_yaml_title(source: str) -> str | None:
    yaml_block = re.match(r'^---\s*\n(.*?)\n---', source, re.DOTALL)
    if not yaml_block:
        return None
    title_line = re.search(r'^title:\s*(.+?)$', yaml_block.group(1), re.MULTILINE)
    return title_line.group(1).strip() if title_line else None


def get_example_headings(qmd_path: Path) -> set[str]:
    """Return the set of heading lines that open an #exm- div block."""
    text = qmd_path.read_text(encoding='utf-8')
    # Match:  ::: {#exm-...}  then optional blank line  then a heading
    pattern = re.compile(
        r'^::: \{#exm-[^}]*\}[^\n]*\n\n?(#{1,6} .+?)(?:\n|$)',
        re.MULTILINE
    )
    return {m.group(1).strip() for m in pattern.finditer(text)}


def prefix_if_example(line: str, example_headings: set[str]) -> str:
    if line.strip() not in example_headings:
        return line
    m = re.match(r'^(#{1,6}\s+)(.*)', line)
    return (m.group(1) + 'Example: ' + m.group(2)) if m else line


def extract_headings(source: str, example_headings: set[str]) -> str:
    lines = source.splitlines()
    result = [
        prefix_if_example(l, example_headings)
        for l in lines if re.match(r'^#{1,6}\s', l)
    ]
    return '\n'.join(result)


def strip_notebook(nb_path: Path, qmd_path: Path) -> None:
    example_headings = get_example_headings(qmd_path) if qmd_path.exists() else set()

    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    for cell in nb['cells']:
        source = ''.join(cell['source'])

        if cell['cell_type'] == 'code':
            new_cells.append(cell)

        elif cell['cell_type'] == 'markdown':
            title = extract_yaml_title(source)
            headings = extract_headings(source, example_headings)

            parts = []
            if title:
                parts.append(f'# {title}')
            if headings:
                parts.append(headings)

            if parts:
                new_cell = {
                    'cell_type': 'markdown',
                    'metadata': cell.get('metadata', {}),
                    'source': ['\n\n'.join(parts)],
                }
                new_cells.append(new_cell)

    nb['cells'] = new_cells
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f'Stripped {nb_path.name}: {len(nb["cells"])} cells, '
          f'{len(example_headings)} example headings found in qmd')
