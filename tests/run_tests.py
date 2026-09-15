"""
run_tests.py
============
Lance toute la suite sans dépendre de pytest.

    python tests/run_tests.py            # tout
    python tests/run_tests.py kernels    # seulement les fichiers dont le nom contient "kernels"

(La suite est également compatible pytest : `pytest tests/` fonctionne si
pytest est installé.)
"""

import importlib.util
import sys
import time
import traceback
from pathlib import Path

import _bootstrap  # noqa: F401

HERE = Path(__file__).resolve().parent


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def main(selector=None):
    sys.path.insert(0, str(HERE))
    files = sorted(p for p in HERE.glob("test_*.py")
                   if selector is None or selector in p.stem)
    passed = failed = 0
    failures = []

    for path in files:
        module = load_module(path)
        names = [n for n in dir(module) if n.startswith("test_")]
        print(f"\n{path.name}  ({len(names)} tests)")
        for name in names:
            t0 = time.perf_counter()
            try:
                getattr(module, name)()
            except Exception:
                failed += 1
                failures.append(f"{path.name}::{name}\n{traceback.format_exc()}")
                print(f"  ECHEC  {name}")
            else:
                passed += 1
                print(f"  ok     {name}  ({time.perf_counter() - t0:.2f} s)")

    print(f"\n{'=' * 60}\n{passed} réussis, {failed} échoués")
    for f in failures:
        print(f"\n{'-' * 60}\n{f}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
