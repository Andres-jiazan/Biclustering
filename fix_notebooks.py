"""Fix all notebooks: replace Chinese comment cells with English, fix newline-in-string issues."""
import json
import re

def fix_source(src):
    """Fix newlines inside string literals by removing them."""
    if isinstance(src, list):
        src = ''.join(src)

    # Replace actual newline (chr(10)) inside string literals with space
    result = []
    in_str = False
    str_char = None
    i = 0
    while i < len(src):
        c = src[i]
        if not in_str:
            if c in ('"', "'"):
                in_str = True
                str_char = c
            result.append(c)
        else:
            if c == str_char:
                in_str = False
                result.append(c)
            elif ord(c) == 10:  # newline inside string - remove it
                result.append(' ')  # replace with space
            else:
                result.append(c)
        i += 1
    return ''.join(result)


def fix_notebook(path):
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = cell['source']
        if isinstance(src, list):
            src = ''.join(src)
        fixed = fix_source(src)
        if fixed != src:
            cell['source'] = fixed
            changed = True

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print(f'Fixed: {path}')
    else:
        print(f'No changes: {path}')
    return changed


# Fix all notebooks
import sys
for path in [
    'notebooks/02_block_reconstruction.ipynb',
    'notebooks/03_nonlinear_transform.ipynb',
    'notebooks/04_full_pipeline.ipynb',
    'notebooks/05_experiments.ipynb',
]:
    fix_notebook(path)
