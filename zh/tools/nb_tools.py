"""
Shared build tooling for the Chinese / modernized-PyMC edition of
"Probabilistic Programming and Bayesian Methods for Hackers".

Each chapter is authored once as a Python list of cell dicts:

    cells = [
        {"type": "markdown", "source": "## 标题\n\n正文 *斜体* **粗体** `code` $x^2$ ..."},
        {"type": "code", "source": "import pymc as pm\n..."},
        ...
    ]

and this module turns that single source of truth into:
  - a runnable .ipynb (via nbformat)
  - a synced .org file (via a small, deliberately-restricted markdown->org
    converter -- restricted because we control the input syntax used across
    every chapter, so we only need to handle the subset we actually write)

The .ipynb is executed separately (jupyter nbconvert --execute) to fill in
outputs; after that, extract_images() below pulls any PNG outputs out of the
executed notebook into files so the .org version can reference the same
images.
"""
import base64
import json
import os
import re

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def write_ipynb(cells, path, kernelspec_name="python3"):
    nb = new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": kernelspec_name},
        "language_info": {"name": "python", "version": "3"},
    }
    for c in cells:
        if c["type"] == "markdown":
            nb["cells"].append(new_markdown_cell(c["source"]))
        elif c["type"] == "code":
            nb["cells"].append(new_code_cell(c["source"]))
        else:
            raise ValueError(f"unknown cell type {c['type']}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


# ---------------------------------------------------------------------
# Minimal, controlled markdown -> org-mode converter.
#
# We only need to support the subset of markdown syntax *we* write across
# these chapters, not arbitrary markdown -- that keeps this reliable.
# Supported: #/##/###/#### headers, **bold**, _italic_, `code`,
# [text](url) links, $inline$ and $$display$$ math, "- " / "1. " lists,
# "> " blockquotes, and "---" horizontal rules.
# ---------------------------------------------------------------------
def md_to_org(text):
    lines = text.split("\n")
    out = []
    in_quote = False
    for line in lines:
        stripped = line.rstrip()

        # headers (the book occasionally uses 5-6 #'s for small run-in
        # headers; org supports arbitrary heading depth via extra *'s)
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            if in_quote:
                out.append("#+END_QUOTE")
                in_quote = False
            level = len(m.group(1))
            out.append("*" * level + " " + _inline_md_to_org(m.group(2)))
            continue

        # horizontal rule
        if re.match(r"^(-{3,}|\*{3,})$", stripped):
            if in_quote:
                out.append("#+END_QUOTE")
                in_quote = False
            out.append("-----")
            continue

        # blockquote
        m = re.match(r"^>\s?(.*)$", stripped)
        if m:
            if not in_quote:
                out.append("#+BEGIN_QUOTE")
                in_quote = True
            out.append(_inline_md_to_org(m.group(1)))
            continue
        else:
            if in_quote:
                out.append("#+END_QUOTE")
                in_quote = False

        # list items (org uses the same "- " / "1. " syntax as markdown)
        out.append(_inline_md_to_org(stripped))

    if in_quote:
        out.append("#+END_QUOTE")
    return "\n".join(out)


def _inline_md_to_org(s):
    # protect $$...$$ and $...$ math spans and `code` spans from further
    # inline substitution by stashing them, then restoring at the end
    stash = []

    def _stash(m):
        stash.append(m.group(0))
        return f"\x00{len(stash)-1}\x00"

    s = re.sub(r"\$\$.*?\$\$", _stash, s)
    s = re.sub(r"\$[^$\n]+\$", _stash, s)
    s = re.sub(r"`[^`\n]+`", lambda m: _stash_code(m, stash), s)

    # links [text](url) -> [[url][text]]
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"[[\2][\1]]", s)
    # bold **text** -> *text*
    s = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", s)
    # italic _text_ -> /text/
    s = re.sub(r"_([^_]+)_", r"/\1/", s)

    for i, val in enumerate(stash):
        if val.startswith("`"):
            val = "=" + val[1:-1] + "="
        s = s.replace(f"\x00{i}\x00", val)
    return s


def _stash_code(m, stash):
    stash.append(m.group(0))
    return f"\x00{len(stash)-1}\x00"


def write_org(cells, path, title, images_by_cell=None):
    """images_by_cell: {code_cell_index (0-based, over ALL cells): [png_path, ...]}
    -- image files to insert immediately after that cell's #+BEGIN_SRC block."""
    images_by_cell = images_by_cell or {}
    lines = [
        f"#+TITLE: {title}",
        "#+LANGUAGE: zh-CN",
        "#+OPTIONS: toc:3 num:2",
        "#+STARTUP: showall",
        "",
    ]
    for idx, c in enumerate(cells):
        if c["type"] == "markdown":
            lines.append(md_to_org(c["source"]))
            lines.append("")
        else:
            lines.append("#+BEGIN_SRC python")
            lines.append(c["source"].rstrip("\n"))
            lines.append("#+END_SRC")
            for img in images_by_cell.get(idx, []):
                lines.append(f"[[file:{img}]]")
            lines.append("")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def extract_images(executed_ipynb_path, out_dir, prefix):
    """Pull image/png outputs out of an executed notebook into files.
    Returns {code_cell_index: [relative_path, ...]} using the SAME cell
    indexing (0-based, over all cells including markdown) as the `cells`
    list used to build the notebook, so callers can line them up with
    write_org()'s images_by_cell argument."""
    os.makedirs(out_dir, exist_ok=True)
    nb = json.load(open(executed_ipynb_path, encoding="utf-8"))
    result = {}
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        pngs = []
        out_i = 0
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            if "image/png" in data:
                b64 = data["image/png"]
                if isinstance(b64, list):
                    b64 = "".join(b64)
                fname = f"{prefix}_{idx:03d}_{out_i}.png"
                with open(os.path.join(out_dir, fname), "wb") as f:
                    f.write(base64.b64decode(b64))
                pngs.append(os.path.join(os.path.basename(out_dir), fname))
                out_i += 1
        if pngs:
            result[idx] = pngs
    return result


def count_errors(executed_ipynb_path):
    nb = json.load(open(executed_ipynb_path, encoding="utf-8"))
    n = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                n += 1
    return n
