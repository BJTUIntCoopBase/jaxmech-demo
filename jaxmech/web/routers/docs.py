"""Documents API — serve markdown files from Documents/."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/docs", tags=["docs"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = PROJECT_ROOT / "Documents"
README = PROJECT_ROOT / "README.md"

# Category metadata: (dir_name, display_label, icon, public)
# public=False → only shown in full installation, excluded from public manifest
_CATEGORIES = [
    ("General",     "总览",     "🌐", True),
    ("Modules",     "模块文档", "🧩", True),
    ("Theory",      "理论参考", "📐", True),
    ("Development", "开发内部", "🛠️", False),
]
_CAT_META = {name: (label, icon, pub) for name, label, icon, pub in _CATEGORIES}


@router.get("")
async def list_docs():
    """List available documentation files grouped by category."""
    docs = []
    # README always first
    if README.is_file():
        docs.append({
            "name": "README.md",
            "rel_path": "README.md",
            "category": "General",
            "category_label": "总览",
            "category_icon": "🌐",
        })
    if not DOCS_DIR.is_dir():
        return docs
    for cat_name, cat_label, cat_icon, _ in _CATEGORIES:
        cat_dir = DOCS_DIR / cat_name
        if not cat_dir.is_dir():
            continue
        for f in sorted(cat_dir.glob("*.md")):
            docs.append({
                "name": f.name,
                "rel_path": f"{cat_name}/{f.name}",
                "category": cat_name,
                "category_label": cat_label,
                "category_icon": cat_icon,
            })
    return docs


@router.get("/{rel_path:path}")
async def get_doc(rel_path: str):
    """Return the raw markdown content of a document.

    *rel_path* is either ``README.md`` or ``<Category>/<filename>.md``.
    """
    if rel_path == "README.md" and README.is_file():
        return {"name": "README.md", "content": README.read_text(encoding="utf-8")}

    doc_path = (DOCS_DIR / rel_path).resolve()
    # Security: must be under DOCS_DIR and be a .md file
    try:
        doc_path.relative_to(DOCS_DIR)
    except ValueError:
        raise HTTPException(403, "Access denied.")
    if doc_path.suffix != ".md" or not doc_path.is_file():
        raise HTTPException(404, f"Document not found: {rel_path}")
    return {"name": doc_path.name, "content": doc_path.read_text(encoding="utf-8")}

