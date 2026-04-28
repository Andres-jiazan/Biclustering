"""Scan notebooks for newlines inside single-quoted string literals."""
import json
import sys

def find_issues(nb_path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    issues = []
    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = cell['source']
        # Scan for newlines inside string literals
        in_str = False
        str_char = None
        i = 0
        while i < len(src):
            c = src[i]
            if not in_str:
                if c in ('"', "'"):
                    in_str = True
                    str_char = c
            else:
                if c == str_char:
                    in_str = False
                elif ord(c) == 10:  # newline inside string
                    ctx_start = max(0, i-30)
                    ctx_end = min(len(src), i+30)
                    issues.append((cell_idx, i, repr(src[ctx_start:ctx_end])))
            i += 1
    return issues

for path in sys.argv[1:]:
    issues = find_issues(path)
    if issues:
        print(f"\n{path}:")
        for cell_idx, pos, ctx in issues:
            print(f"  Cell {cell_idx}, pos {pos}: {ctx}")
    else:
        print(f"{path}: OK")
