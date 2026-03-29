from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "overlay_algebra"
_SRC_INIT = _SRC_PACKAGE_DIR / "__init__.py"

if not _SRC_INIT.is_file():
    raise ModuleNotFoundError(
        "Expected src/overlay_algebra/__init__.py to exist for the M0 bootstrap package."
    )

# Bootstrap the real src-layout package so plain `python -c "import overlay_algebra"`
# works from the repo root before editable-install packaging is in place.
__file__ = str(_SRC_INIT)
__path__ = [str(_SRC_PACKAGE_DIR)]

with _SRC_INIT.open("r", encoding="utf-8") as handle:
    exec(compile(handle.read(), __file__, "exec"), globals(), globals())
