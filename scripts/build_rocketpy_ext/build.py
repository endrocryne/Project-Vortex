"""
Build script — packs rocketpy_integration/ into a .vortexext file.

Usage:
    cd scripts/build_rocketpy_ext
    python build.py
"""

import os
import sys

# Add the project root so we can import the Vortex packaging module.
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from extensions.packaging import pack_extension

_here = os.path.dirname(os.path.abspath(__file__))
_source_dir = os.path.join(_here, "rocketpy_integration")
_output = os.path.join(_here, "rocketpy_integration.vortexext")


def main():
    print(f"Source : {_source_dir}")
    print(f"Output : {_output}")
    result = pack_extension(_source_dir, _output)
    size_kb = os.path.getsize(result) / 1024
    print(f"Packaged successfully → {result}  ({size_kb:.1f} KB)")
    return result


if __name__ == "__main__":
    main()
