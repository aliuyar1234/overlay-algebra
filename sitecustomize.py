from __future__ import annotations

import sys
from pathlib import Path


# Allow `python -c "import overlay_algebra"` from the repo root without
# requiring an editable install during the early scaffold milestones.
SRC_DIR = Path(__file__).resolve().parent / "src"
if SRC_DIR.is_dir():
    src_text = str(SRC_DIR)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
