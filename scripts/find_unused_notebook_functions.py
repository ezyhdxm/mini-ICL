"""
Find top-level functions in icl (excluding legacy paths) not referenced from notebooks.

Resolution strategy:
- Enumerate module-level def/async def in each non-legacy .py file under src/icl.
- Parse all target notebooks: Import/ImportFrom, and track import aliases for attribute calls.
- Resolve (import_module, name) to a defining module by following relative imports in
  __init__.py and .py files (one level of 'from .x import y' chains).
- ``from M import *`` is expanded using ``M``'s explicit re-exports and ``__all__`` when it is a
  static list in ``M``'s AST (e.g. ``icl.coin.coin_analysis`` re-exporting ``icl.coin.analysis``).
- Mark a definition (defining_module, func_name) as used if any notebook import resolves to it.

Run from repo root: python scripts/find_unused_notebook_functions.py
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
SRC_ICL = ROOT / "src" / "icl"
NOTEBOOK_DIR = ROOT / "notebooks"

# Notebooks to scan (exclude *_legacy.ipynb)
NOTEBOOK_GLOB = "*.ipynb"


def is_legacy_path(path: Path) -> bool:
    """Skip directories (path segments) named ``legacy``, not e.g. ``legacy.py`` filenames."""
    parts = {p.lower() for p in path.parts}
    return "legacy" in parts


def iter_icl_py_files() -> List[Path]:
    out = []
    for p in SRC_ICL.rglob("*.py"):
        if is_legacy_path(p):
            continue
        out.append(p)
    return sorted(out)


def get_top_level_functions(tree: ast.AST) -> List[str]:
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.append(node.name)
    return names


def parse_file(path: Path) -> Optional[ast.AST]:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


# --- resolution cache: module_id -> parsed body imports ---

def module_id_from_path(py_path: Path) -> Optional[str]:
    """Path under src/ -> dotted module (icl.foo.bar)."""
    try:
        rel = py_path.relative_to(SRC_ICL.parent)
    except ValueError:
        return None
    if rel.name == "__init__.py":
        return ".".join(rel.parent.parts) if rel.parent.parts else None
    return ".".join(rel.with_suffix("").parts)


def resolve_relative_import_file(parent_file: Path, level: int, module_suffix: str) -> Optional[Path]:
    """Map ImportFrom(level, module) to a file path."""
    parent_dir = parent_file.parent
    if level == 0:
        if not module_suffix:
            return None
        parts = module_suffix.split(".")
        cand = SRC_ICL.parent / Path(*parts) / "__init__.py"
        if cand.exists():
            return cand
        cand = SRC_ICL.parent / Path(*parts).with_suffix(".py")
        if cand.exists():
            return cand
        return None
    # relative
    base = parent_dir
    for _ in range(level - 1):
        base = base.parent
    if not module_suffix:
        init_py = base / "__init__.py"
        return init_py if init_py.exists() else None
    parts = module_suffix.split(".")
    sub = base.joinpath(*parts)
    init_py = sub / "__init__.py"
    if init_py.exists():
        return init_py
    py = sub.with_suffix(".py")
    if py.exists():
        return py
    return None


def _const_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # Python 3.7 compatibility
    if isinstance(node, ast.Str):  # type: ignore[attr-defined]
        return node.s
    return None


def parse___all___names(tree: ast.AST) -> Optional[List[str]]:
    """If ``__all__`` is assigned a static list/tuple of string literals, return those names."""
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    names: List[str] = []
                    for elt in node.value.elts:
                        s = _const_str(elt)
                        if s is None:
                            return None
                        names.append(s)
                    return names
                return None
    return None


def _module_path_for_id(module_id: str) -> Optional[Path]:
    parts = module_id.split(".")
    base = SRC_ICL.parent.joinpath(*parts)
    init_py = base / "__init__.py"
    mod_py = base.with_suffix(".py")
    if init_py.exists():
        return init_py
    if mod_py.exists():
        return mod_py
    return None


def expand_star_import(module_id: str) -> Dict[str, Tuple[str, str]]:
    """
    Approximate names bound by ``from module_id import *``: explicit ImportFrom bindings in
    ``module_id``'s module file, filtered to ``__all__`` when it is a static list.
    """
    path = _module_path_for_id(module_id)
    if not path:
        return {}
    tree = parse_file(path)
    if not tree:
        return {}
    raw = collect_reexports_from_tree(tree, path)
    all_names = parse___all___names(tree)
    if all_names is not None:
        return {n: raw[n] for n in all_names if n in raw}
    return dict(raw)


def collect_reexports_from_tree(
    tree: ast.AST, parent_file: Path
) -> Dict[str, Tuple[str, str]]:
    """
    Map symbol -> (defining_module_id, original_name) for names bound in this file
    via imports (including 'from .x import y').
    """
    out: Dict[str, Tuple[str, str]] = {}
    parent_mod = module_id_from_path(parent_file)
    if not parent_mod:
        return out

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            mod_name = node.module
            if node.level and node.level > 0:
                resolved = resolve_relative_import_file(parent_file, node.level, mod_name or "")
                if resolved:
                    sub_mod = module_id_from_path(resolved)
                else:
                    sub_mod = None
            else:
                sub_mod = mod_name
            if not sub_mod:
                continue
            for alias in node.names:
                if alias.name == "*":
                    for local, target in expand_star_import(sub_mod).items():
                        out[local] = target
                    continue
                local = alias.asname or alias.name
                out[local] = (sub_mod, alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[-1]
                out[local] = (alias.name, alias.name.split(".")[-1])
    return out


# Cache: module_id -> dict local_name -> (defining_module, orig_name)
_reexport_cache: Dict[str, Dict[str, Tuple[str, str]]] = {}


def load_module_reexports(module_id: str) -> Dict[str, Tuple[str, str]]:
    if module_id in _reexport_cache:
        return _reexport_cache[module_id]
    parts = module_id.split(".")
    base = SRC_ICL.parent.joinpath(*parts)
    init_py = base / "__init__.py"
    mod_py = base.with_suffix(".py")
    path = init_py if init_py.exists() else (mod_py if mod_py.exists() else None)
    if not path:
        _reexport_cache[module_id] = {}
        return {}
    tree = parse_file(path)
    if not tree:
        _reexport_cache[module_id] = {}
        return {}
    d = collect_reexports_from_tree(tree, path)
    _reexport_cache[module_id] = d
    return d


def defining_module_for_symbol(module_id: str, symbol: str, _depth: int = 0) -> Optional[str]:
    """Resolve where `symbol` is defined when accessed as module_id.symbol."""
    if _depth > 30:
        return None
    path_parts = module_id.split(".")
    base = SRC_ICL.parent.joinpath(*path_parts)
    py_file = base.with_suffix(".py")
    if py_file.exists():
        tree = parse_file(py_file)
        if tree and symbol in get_top_level_functions(tree):
            return module_id
    init_py = base / "__init__.py"
    if init_py.exists():
        tree = parse_file(init_py)
        if tree and symbol in get_top_level_functions(tree):
            return module_id
    re = load_module_reexports(module_id)
    if symbol in re:
        sub_mod, orig = re[symbol]
        if sub_mod == module_id:
            return None
        return defining_module_for_symbol(sub_mod, orig, _depth + 1)
    return None


_IPY_MAGIC = re.compile(r"^\s*[%!]")


def _sanitize_cell_for_ast(src: str) -> str:
    """Drop IPython magics / shell lines so ast.parse succeeds."""
    lines = []
    for line in src.splitlines():
        if _IPY_MAGIC.match(line):
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_notebook_imports_and_uses(
    nb_path: Path,
) -> Tuple[Set[Tuple[str, str]], Dict[str, Tuple[str, str]]]:
    data = json.loads(nb_path.read_text(encoding="utf-8"))
    imports: Set[Tuple[str, str]] = set()
    alias_to_module: Dict[str, str] = {}
    from_bindings: Dict[str, Tuple[str, str]] = {}

    trees: List[ast.AST] = []
    for cell in data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        raw = "".join(cell.get("source", []))
        cleaned = _sanitize_cell_for_ast(raw)
        if not cleaned.strip():
            continue
        try:
            trees.append(ast.parse(cleaned))
        except SyntaxError:
            continue

    def collect_imports(tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    full = alias.name
                    if alias.asname:
                        alias_to_module[alias.asname] = full
                    else:
                        parts = full.split(".")
                        alias_to_module[parts[-1]] = full
            elif isinstance(node, ast.ImportFrom):
                if node.module is None:
                    continue
                mod = node.module
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local = alias.asname or alias.name
                    imports.add((mod, alias.name))
                    from_bindings[local] = (mod, alias.name)

    for t in trees:
        collect_imports(t)

    attr_refs: Set[Tuple[str, str]] = set()

    def resolve_alias_base(name: str) -> Optional[str]:
        if name in from_bindings:
            return from_bindings[name][0]
        if name in alias_to_module:
            return alias_to_module[name]
        return None

    def collect_calls(tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                if isinstance(f, ast.Attribute):
                    if isinstance(f.value, ast.Name):
                        base_mod = resolve_alias_base(f.value.id)
                        if base_mod:
                            attr_refs.add((base_mod, f.attr))

    for t in trees:
        collect_calls(t)

    imports |= attr_refs
    return imports, from_bindings


def main() -> None:
    definitions: Dict[Tuple[str, str], str] = {}
    for py in iter_icl_py_files():
        tree = parse_file(py)
        if not tree:
            continue
        mid = module_id_from_path(py)
        if not mid:
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = (ast.get_docstring(node) or "").strip().split("\n")[0][:400]
            definitions[(mid, node.name)] = doc

    notebooks = sorted(
        p
        for p in NOTEBOOK_DIR.glob(NOTEBOOK_GLOB)
        if p.is_file() and "legacy" not in p.name.lower()
    )
    used_defs: Set[Tuple[str, str]] = set()
    for nb in notebooks:
        refs, _ = extract_notebook_imports_and_uses(nb)
        for mod, name in refs:
            dm = defining_module_for_symbol(mod, name)
            if dm:
                used_defs.add((dm, name))

    unused = sorted(set(definitions.keys()) - used_defs, key=lambda x: (x[0], x[1]))

    def icl_subpackage(mid: str) -> str:
        parts = mid.split(".")
        if len(parts) >= 2 and parts[0] == "icl":
            return parts[1]
        return "other"

    counts = Counter(icl_subpackage(mid) for mid, _ in unused)
    by_area: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for mid, name in unused:
        by_area[icl_subpackage(mid)].append((mid, name))

    out_lines = [
        "# Non-legacy `icl` functions unused by notebooks",
        "",
        "Generated by static analysis: top-level `def` / `async def` in `src/icl` (skipping directories named `legacy` in the path),",
        "minus symbols reachable from imports in `notebooks/*.ipynb` (excluding `*_legacy.ipynb`).",
        "Resolution follows `from … import …` re-exports in `__init__.py` / `.py` files. Library-side `from M import *` is expanded",
        "using explicit imports in `M` filtered by a static `__all__` list when present (covers `icl.coin.coin_analysis`).",
        "Notebook star imports and dynamic imports are not modeled; other edge cases may still yield false positives or negatives.",
        "",
        "**Regenerate:** `python scripts/find_unused_notebook_functions.py` from the repo root.",
        "",
        "**Scope:** Module-level functions only (not class methods). Internal callers and CLI/tests outside notebooks",
        "still count as unused for this report.",
        "",
        f"**Notebooks scanned:** {', '.join(p.name for p in notebooks)}",
        "",
        "**Note:** On case-insensitive filesystems, `coins.ipynb` may be listed with different casing.",
        "",
        f"**Total unused (module-level) functions:** {len(unused)}",
        "",
        "## Counts by `icl` subpackage",
        "",
        "| Subpackage | Unused functions |",
        "|------------|------------------|",
    ]
    for subpkg in sorted(counts.keys(), key=lambda s: (-counts[s], s)):
        out_lines.append(f"| `{subpkg}` | {counts[subpkg]} |")
    out_lines.extend(["", "---", ""])

    for subpkg in sorted(by_area.keys(), key=lambda s: (-len(by_area[s]), s)):
        out_lines.append(f"## `icl.{subpkg}`")
        out_lines.append("")
        for mid, name in sorted(by_area[subpkg], key=lambda x: (x[0], x[1])):
            doc = definitions.get((mid, name), "")
            desc = doc if doc else "*No docstring.*"
            out_lines.append(f"- **`{mid}.{name}`** — {desc}")
        out_lines.append("")

    out_path = ROOT / "notebooks" / "NOTEBOOK_UNUSED_FUNCTIONS.md"
    out_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(unused)} unused)")


if __name__ == "__main__":
    main()
